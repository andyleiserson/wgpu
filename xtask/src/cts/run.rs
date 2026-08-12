//! Check out, build, and run CTS tests.

use anyhow::{anyhow, bail, Context};
use core::fmt;
use pico_args::Arguments;
use regex_lite::Regex;
use std::ffi::OsString;
use xshell::Shell;

use crate::cts::{
    build_cts_runner, ensure_cts_checkout, parse_lst_line, CTS_BIN_ARGS, CTS_DEFAULT_TEST_LIST,
};

#[derive(Default)]
struct TestLine {
    pub selector: OsString,
    pub fails_if: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
enum PrintOutputWhen {
    TestFails,
    Always,
}

/// `stray_positional` is a leading positional argument that the sub-subcommand
/// dispatcher had to consume from `args` in order to inspect it. It is a test
/// selector, and belongs at the front of the list.
pub fn run_cts(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
    stray_positional: Option<OsString>,
) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");
    let llvm_cov = args.contains("--llvm-cov");
    let release = args.contains("--release");

    // This is used in the Vulkan hal to waive pre-existing validation
    // errors in the CTS, until they can be fixed.
    shell.set_var("WGPU_CTS_XTASK", "1");

    let output_filter = args
        .opt_value_from_str::<_, String>("--print-output-when")?
        .map(|f| {
            let values = [
                ("test-fails", PrintOutputWhen::TestFails),
                ("always", PrintOutputWhen::Always),
            ];
            let lowered = f.to_ascii_lowercase();
            values
                .iter()
                .find_map(|(cli_str, enum_value)| (&*lowered == *cli_str).then_some(*enum_value))
                .ok_or_else(|| {
                    anyhow!(
                        "`{f}` is not a valid `--print-output-when` value; expected one of {}",
                        fmt::from_fn(|f| {
                            f.debug_list()
                                .entries(values.iter().map(|(cli, _enum)| cli))
                                .finish()
                        })
                    )
                })
        })
        .transpose()?;

    let running_on_backend = args.opt_value_from_str::<_, String>("--backend")?;
    let enable_external_texture = args.contains("--enable-external-texture")
        || (!args.contains("--disable-external-texture")
            && running_on_backend
                .as_ref()
                .is_some_and(|b| ["metal", "dx12"].contains(&b.as_str())));

    let mut filter_pattern = args.opt_value_from_str::<_, String>("--filter")?;
    let mut filter_invert = false;

    if let Some(filter) = filter_pattern.as_deref() {
        if let Some(filter) = filter.strip_prefix('!') {
            filter_pattern = Some(filter.to_owned());
            filter_invert = true;
        }
    }

    // Compile filter regex early to fail fast on invalid patterns
    let filter = if let Some(pattern) = filter_pattern {
        Some(
            Regex::new(&pattern)
                .context(format!("Invalid regex pattern '{pattern}' for --filter"))?,
        )
    } else {
        None
    };

    let mut list_files = Vec::<OsString>::new();
    while let Some(file) = args.opt_value_from_str("-f")? {
        list_files.push(file);
    }

    let mut tests = stray_positional
        .into_iter()
        .chain(args.finish())
        .map(|selector| TestLine {
            selector,
            ..Default::default()
        })
        .collect::<Vec<_>>();

    if let Some(backend) = &running_on_backend {
        shell.set_var("DENO_WEBGPU_BACKEND", backend);
    } else if !list_files.is_empty() || tests.is_empty() {
        log::warn!("The `--backend` option was not provided. `fails-if` conditions and external");
        log::warn!("texture support are handled correctly only when a backend is specified.");
    }

    #[cfg(windows)]
    if running_on_backend.as_ref().is_none_or(|b| b == "dx12") {
        const DENO_WEBGPU_DX12_COMPILER: &str = "DENO_WEBGPU_DX12_COMPILER";
        const DEFAULT_DX12_COMPILER: &str = "dynamicdxc";

        match shell.var(DENO_WEBGPU_DX12_COMPILER) {
            Ok(value) => {
                log::info!("Using `{DENO_WEBGPU_DX12_COMPILER}` = {value:?} from environment")
            }
            Err(_) => {
                shell.set_var(DENO_WEBGPU_DX12_COMPILER, DEFAULT_DX12_COMPILER);
                log::info!(
                    "Using default `{DENO_WEBGPU_DX12_COMPILER}` = {DEFAULT_DX12_COMPILER:?}"
                );
            }
        }
    }

    let mut default_output_filter = PrintOutputWhen::Always;

    if tests.is_empty() && list_files.is_empty() {
        if passthrough_args.is_none() {
            log::info!("Reading default test list from {CTS_DEFAULT_TEST_LIST}");
            list_files.push(OsString::from(CTS_DEFAULT_TEST_LIST));

            default_output_filter = PrintOutputWhen::TestFails;
        }
    } else if passthrough_args.is_some() {
        bail!("Test(s) and test list(s) are incompatible with passthrough arguments.");
    }

    let output_filter = output_filter.unwrap_or(default_output_filter);

    for file in list_files {
        tests.extend(
            shell
                .read_file(file)?
                .lines()
                .filter_map(parse_lst_line)
                .map(|line| TestLine {
                    selector: OsString::from(line.selector),
                    fails_if: line.fails_if,
                }),
        )
    }

    // Apply filter if specified
    if let Some(ref filter) = filter {
        let original_count = tests.len();
        tests.retain(|test| {
            let selector_str = test.selector.to_string_lossy();
            let matched = filter.is_match(&selector_str);
            if filter_invert {
                !matched
            } else {
                matched
            }
        });
        let filtered_count = tests.len();
        if filtered_count == original_count {
            log::warn!("Filter did not exclude any tests");
        } else if filtered_count != 0 {
            log::info!(
                "Filter selected {filtered_count} of {original_count} test{}",
                if original_count == 1 { "" } else { "s" },
            );
        } else if filtered_count == 0 {
            bail!("Filter did not select any tests");
        } else {
            bail!("Filtering introduced additional tests??");
        }
    }

    let wgpu_cargo_toml = ensure_cts_checkout(&shell, skip_checkout)?;
    let (bin, env_vars) = build_cts_runner(&shell, &wgpu_cargo_toml, release, llvm_cov)?;

    if let Some(passthrough_args) = passthrough_args {
        return Ok(shell
            .cmd(bin)
            .envs(env_vars)
            .args(CTS_BIN_ARGS)
            .args(enable_external_texture.then_some("--enable-external-texture"))
            .args(&passthrough_args)
            .run()?);
    }

    log::info!("Running CTS");
    for test in &tests {
        if let Some(running_on_backend) = &running_on_backend {
            if test.fails_if.contains(running_on_backend) {
                log::info!(
                    "Skipping {} on {} backend",
                    test.selector.to_string_lossy(),
                    running_on_backend,
                );
                continue;
            }
        }

        if let PrintOutputWhen::Always = output_filter {
            log::info!("Running {}", test.selector.to_string_lossy());
        }

        let cmd = shell
            .cmd(&bin)
            .envs(env_vars.clone())
            .args(enable_external_texture.then_some("--enable-external-texture"))
            .args(CTS_BIN_ARGS)
            .args([&test.selector]);

        match output_filter {
            PrintOutputWhen::TestFails => {
                let output = cmd.ignore_status().output().context("Failed to run CTS")?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    if let Some((_, summary)) = stdout.split_once("** Summary **") {
                        println!("\n== Summary for {} ==", test.selector.to_string_lossy());
                        println!("{}", summary.trim());
                    } else {
                        log::info!("Running {}", test.selector.to_string_lossy());
                        print!("{}", stdout);
                        eprint!("{}", stderr);
                    }
                } else {
                    log::info!("Running {}", test.selector.to_string_lossy());
                    print!("{}", stdout);
                    eprint!("{}", stderr);
                    bail!("CTS failed ({})", output.status);
                }
            }
            PrintOutputWhen::Always => {
                cmd.run().context("CTS failed")?;
            }
        }
    }

    if tests.len() > 1 {
        log::info!("Summary reflects only tests from the last selector, not the entire run.");
    }

    Ok(())
}
