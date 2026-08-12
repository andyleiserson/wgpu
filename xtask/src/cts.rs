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

use pico_args::Arguments;
use std::ffi::OsString;
use xshell::Shell;

/// Path within the repository where the CTS will be checked out.
const CTS_CHECKOUT_PATH: &str = "cts";

/// Path within the repository to a file containing the git revision of the CTS to check out.
const CTS_REVISION_PATH: &str = "cts_runner/revision.txt";

/// URL of the CTS git repository.
const CTS_GIT_URL: &str = "https://github.com/gpuweb/cts.git";

/// Path to default CTS test list.
const CTS_DEFAULT_TEST_LIST: &str = "cts_runner/test.lst";

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
            anyhow::bail!("`cts verify-selectors` is not implemented yet")
        }
    }
}

enum Subcommand {
    Run,
    VerifySelectors,
}
