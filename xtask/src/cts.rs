//! Interface for working with the WebGPU CTS (Conformance Test Suite).
//!
//! There are two sub-subcommands, [`run`](run::run_cts) and
//! [`verify-selectors`](verify_selectors::verify_selectors). `run` is the
//! default, so `cargo xtask cts <args>` means `cargo xtask cts run <args>`.
//!
//! To run the default set of tests from `cts_runner/test.lst`:
//!
//! ```sh
//! cargo xtask cts
//! ```
//!
//! To run a specific test selector:
//!
//! ```sh
//! cargo xtask cts 'webgpu:api,operation,command_buffer,basic:*'
//! ```
//!
//! You can also supply your own test list in a file:
//!
//! ```sh
//! cargo xtask cts -f your_tests.lst
//! ```
//!
//! Each line in a test list file is a test selector that will be passed to the
//! CTS's own command line runner. Note that wildcards may only be used to specify
//! running all tests in a file, or all subtests in a test.
//!
//! A test line may optionally contain a `fails-if(backend)` clause. This
//! indicates that the test should be skipped on that backend, however, the
//! runner will only do so if the `--backend` flag is passed to tell it where
//! it is running.
//!
//! Lines starting with `//` or `#` in the test list are treated as comments and
//! ignored.

mod run;
mod verify_selectors;

use anyhow::{bail, Context};
use pico_args::Arguments;
use regex_lite::Regex;
use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    sync::LazyLock,
};
use xshell::Shell;

use crate::util::{git_version_at_least, parse_binary_from_cargo_json};

/// Path within the repository where the CTS will be checked out.
const CTS_CHECKOUT_PATH: &str = "cts";

/// Path within the repository to a file containing the git revision of the CTS to check out.
const CTS_REVISION_PATH: &str = "cts_runner/revision.txt";

/// URL of the CTS git repository.
const CTS_GIT_URL: &str = "https://github.com/gpuweb/cts.git";

/// Path to default CTS test list.
const CTS_DEFAULT_TEST_LIST: &str = "cts_runner/test.lst";

/// Arguments that invoke the CTS's own command line runner, to be passed to the
/// `cts_runner` binary. Relative to the CTS checkout, which is where
/// [`ensure_cts_checkout`] leaves the shell.
pub const CTS_BIN_ARGS: [&str; 2] = ["./tools/run_deno", "--verbose"];

pub fn run(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    // `run` is the default sub-subcommand, so a leading positional that isn't a
    // known sub-subcommand is a test selector. `pico-args` has already consumed
    // it and offers no way to put it back, so carry it forward explicitly.
    let (subcommand, stray_positional) = match args.subcommand()?.as_deref() {
        Some("run") => (Subcommand::Run, None),
        Some("verify-selectors") => (Subcommand::VerifySelectors, None),
        Some(selector) => (Subcommand::Run, Some(OsString::from(selector))),
        None => (Subcommand::Run, None),
    };

    match subcommand {
        Subcommand::Run => run::run_cts(shell, args, passthrough_args, stray_positional),
        Subcommand::VerifySelectors => {
            if passthrough_args.is_some() {
                bail!("`cts verify-selectors` does not accept passthrough arguments.");
            }
            verify_selectors::verify_selectors(shell, args, stray_positional)
        }
    }
}

enum Subcommand {
    Run,
    VerifySelectors,
}

/// A selector-bearing line parsed from a `.lst` test list file.
struct LstLine {
    selector: String,
    /// Backends named in a `fails-if(...)` clause, empty if there was none.
    fails_if: Vec<String>,
}

/// Parse one line of a `.lst` file, returning `None` for blank lines and lines
/// that hold nothing but a comment.
///
/// Both `#` and `//` start a comment, which may either occupy the whole line or
/// follow a selector on the same line.
fn parse_lst_line(line: &str) -> Option<LstLine> {
    static FAILS_IF_REGEX: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^fails-if\s*\(\s*(\w+(?:,\w+)*)\s*\)\s+").unwrap());

    let line = line.trim_start();
    let line = line.split('#').next().unwrap();
    if line.starts_with("//") {
        return None;
    }

    let (fails_if, rest) = match FAILS_IF_REGEX.captures(line) {
        Some(captures) => {
            let backends = captures[1].split(',').map(str::to_string).collect();
            (backends, &line[captures[0].len()..])
        }
        None => (Vec::new(), line),
    };

    let selector = rest.split("//").next().unwrap().trim();

    (!selector.is_empty()).then(|| LstLine {
        selector: selector.to_string(),
        fails_if,
    })
}

/// Check out the pinned CTS revision, cloning the CTS first if it isn't present.
///
/// Returns the absolute path to the workspace `Cargo.toml`. That path has to be
/// resolved before we move, because this changes `shell`'s working directory to
/// the CTS checkout and leaves it there for the caller.
pub fn ensure_cts_checkout(shell: &Shell, skip_checkout: bool) -> anyhow::Result<PathBuf> {
    let wgpu_cargo_toml = std::path::absolute(shell.current_dir().join("Cargo.toml"))
        .context("Failed to get path to `Cargo.toml`")?;

    let cts_revision = shell
        .read_file(CTS_REVISION_PATH)
        .context(format!(
            "Failed to read CTS git SHA from `{CTS_REVISION_PATH}`"
        ))?
        .trim()
        .to_string();

    if !shell.path_exists(CTS_CHECKOUT_PATH) {
        if skip_checkout {
            bail!("Skipping CTS checkout doesn't make sense when CTS is not present");
        }
        let mut cmd = shell
            .cmd("git")
            .args(["clone", CTS_GIT_URL, CTS_CHECKOUT_PATH])
            .quiet();

        if git_version_at_least(shell, [2, 49, 0])? {
            log::info!("Cloning CTS shallowly with revision {cts_revision}");
            cmd = cmd.args(["--depth=1", "--revision", &cts_revision]);
            cmd = cmd.args([
                "-c",
                "remote.origin.fetch=+refs/heads/gh-pages:refs/remotes/origin/gh-pages",
                "-c",
                "advice.detachedHead=false",
            ]);
        } else {
            log::info!("Cloning full checkout of CTS with revision {cts_revision}");
            cmd = cmd.args(["-b", "gh-pages", "--single-branch"]);
        }

        cmd.run().context("Failed to clone CTS")?;

        shell.change_dir(CTS_CHECKOUT_PATH);
    } else if !skip_checkout {
        shell.change_dir(CTS_CHECKOUT_PATH);

        // For new clones, this is set by the cloning commands above, but older
        // clones may not have it. Eventually this can be removed.
        if shell
            .cmd("git")
            .args(["config", "--get", "remote.origin.fetch"])
            .quiet()
            .ignore_stdout()
            .ignore_stderr()
            .run()
            .is_err()
        {
            shell
                .cmd("git")
                .args([
                    "config",
                    "remote.origin.fetch",
                    "+refs/heads/gh-pages:refs/remotes/origin/gh-pages",
                ])
                .quiet()
                .run()
                .context("Failed setting git config")?;
        }

        // If we don't have the CTS commit we want, try to fetch it.
        if shell
            .cmd("git")
            .args(["cat-file", "commit", &cts_revision])
            .quiet()
            .ignore_stdout()
            .ignore_stderr()
            .run()
            .is_err()
        {
            log::info!("Fetching CTS");
            shell
                .cmd("git")
                .args(["fetch", "--quiet"])
                .quiet()
                .run()
                .context("Failed to fetch CTS")?;
        }
    } else {
        shell.change_dir(CTS_CHECKOUT_PATH);
    }

    if !skip_checkout {
        log::info!("Checking out CTS");
        shell
            .cmd("git")
            .args(["checkout", "--quiet", &cts_revision])
            .quiet()
            .run()
            .context("Failed to check out CTS")?;
    } else {
        log::info!("Skipping CTS checkout because --skip-checkout was specified");
    }

    Ok(wgpu_cargo_toml)
}

/// Build the `cts_runner` binary, returning its path along with the environment
/// variables that must be set when running it. The environment is empty unless
/// `llvm_cov` is set.
pub fn build_cts_runner(
    shell: &Shell,
    wgpu_cargo_toml: &Path,
    release: bool,
    llvm_cov: bool,
) -> anyhow::Result<(String, Vec<(String, String)>)> {
    let mut cargo_opts: Vec<OsString> = vec![
        "--manifest-path".into(),
        wgpu_cargo_toml.into(),
        "-p".into(),
        "cts_runner".into(),
        "--bin".into(),
        "cts_runner".into(),
    ];
    if release {
        cargo_opts.push("--release".into());
    }

    let env_vars = if llvm_cov && env::var("CARGO_LLVM_COV_SHOW_ENV").is_err() {
        // Typically coverage runs are done via cargo with `cargo llvm-cov run`, but we want
        // to run the coverage-instrumented binary directly because that is much faster than
        // invoking `cargo` repeatedly for each CTS test selector. Running it directly requires
        // setting some environment variables output by `cargo llvm-cov show-env`. See
        // <https://github.com/taiki-e/cargo-llvm-cov/blob/main/README.md#get-coverage-of-external-tests>
        //
        // In CI the variables are set by the GitHub workflow, which we detect by checking
        // whether `CARGO_LLVM_COV_SHOW_ENV` is already set. If the environment variables
        // have not been set already, query them now so we can add them to the commands
        // we run.
        let env = shell
            .cmd("cargo")
            .args(["llvm-cov", "--no-cfg-coverage", "show-env"])
            .read()
            .context("Failed to get `llvm-cov` environment variables")?
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    None
                } else {
                    line.split_once('=')
                }
            })
            .map(|(key, value)| {
                let value = value.trim_matches('"').trim_matches('\'');
                (key.to_string(), value.to_string())
            })
            .collect::<Vec<_>>();

        // Avoid conflicts between coverage and non-coverage build artifacts.
        // This is recommended by the `cargo-llvm-cov` docs.
        shell
            .cmd("cargo")
            .envs(env.clone())
            .args(["llvm-cov", "clean", "--workspace"])
            .run()
            .context("Failed to run `llvm-cov clean`")?;

        env
    } else {
        vec![]
    };

    let build_output = shell
        .cmd("cargo")
        .envs(env_vars.clone())
        .args(["build", "--message-format", "json-render-diagnostics"])
        .args(&cargo_opts)
        .read()
        .context("Failed to build `cts_runner`")?;

    let bin = parse_binary_from_cargo_json(&build_output)
        .context("Failed to identify executable from cargo build output")?;

    Ok((bin, env_vars))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Option<(String, Vec<String>)> {
        parse_lst_line(line).map(|l| (l.selector, l.fails_if))
    }

    fn selector(line: &str) -> String {
        parse(line).expect("expected a selector").0
    }

    #[test]
    fn test_parse_lst_line_skips_non_selectors() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("   "), None);
        assert_eq!(parse("// a comment"), None);
        assert_eq!(parse("  // an indented comment"), None);
        assert_eq!(parse("# a comment"), None);
        assert_eq!(parse("  # an indented comment"), None);
    }

    #[test]
    fn test_parse_lst_line_selectors() {
        assert_eq!(
            selector("webgpu:api,validation,buffer,mapping:*"),
            "webgpu:api,validation,buffer,mapping:*"
        );
        assert_eq!(selector("  webgpu:foo:*  "), "webgpu:foo:*");
        assert_eq!(selector("unittests:*"), "unittests:*");
    }

    #[test]
    fn test_parse_lst_line_strips_inline_comments() {
        assert_eq!(selector("webgpu:foo:* // crash"), "webgpu:foo:*");
        assert_eq!(selector("webgpu:foo:*// crash"), "webgpu:foo:*");
        assert_eq!(selector("webgpu:foo:* # crash"), "webgpu:foo:*");
        // Only the first `//` starts the comment, so URLs within it are harmless.
        assert_eq!(
            selector("webgpu:foo:* // see https://github.com/gfx-rs/wgpu/issues/9455"),
            "webgpu:foo:*"
        );
        // `#` is stripped before `//`, so a `#` inside a comment is also harmless.
        assert_eq!(
            selector("webgpu:foo:* // missing const eval (#4507)"),
            "webgpu:foo:*"
        );
    }

    #[test]
    fn test_parse_lst_line_fails_if() {
        assert_eq!(
            parse("fails-if(vulkan) webgpu:foo:*"),
            Some(("webgpu:foo:*".into(), vec!["vulkan".into()]))
        );
        assert_eq!(
            parse("fails-if(dx12,vulkan,metal) webgpu:foo:*"),
            Some((
                "webgpu:foo:*".into(),
                vec!["dx12".into(), "vulkan".into(), "metal".into()]
            ))
        );
        // Whitespace around the parentheses is tolerated.
        assert_eq!(
            parse("fails-if ( vulkan )  webgpu:foo:*"),
            Some(("webgpu:foo:*".into(), vec!["vulkan".into()]))
        );
        assert_eq!(
            parse("fails-if(vulkan) webgpu:foo:* // and it crashes"),
            Some(("webgpu:foo:*".into(), vec!["vulkan".into()]))
        );
        // A `fails-if` clause with no selector after it is not a selector line.
        assert_eq!(parse("fails-if(vulkan) // nothing here"), None);
    }
}
