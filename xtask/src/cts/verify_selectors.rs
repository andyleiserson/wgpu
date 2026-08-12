//! Verify that every CTS selector is accounted for by exactly one line in the
//! `.lst` files.
//!
//! A selector that no line matches has been added to the CTS without anyone
//! deciding whether wgpu passes it; a selector that several lines match has an
//! ambiguous expectation. Both are reported, along with a minimal set of
//! wildcard patterns covering them, which can be pasted into a `.lst` file.

use anyhow::{bail, Context};
use pico_args::Arguments;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    io::IsTerminal,
};
use xshell::Shell;

use crate::cts::{
    build_cts_runner, ensure_cts_checkout, parse_lst_line, LstLine, CTS_BIN_ARGS,
    CTS_DEFAULT_TEST_LIST,
};

/// Selectors verified when none are given on the command line.
const DEFAULT_SELECTORS: [&str; 2] = ["webgpu:api,validation,*", "webgpu:shader,validation,*"];

/// The `.lst` files a selector may be listed in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LstFile {
    Test,
    Fail,
    Skip,
}

impl LstFile {
    const ALL: [LstFile; 3] = [LstFile::Test, LstFile::Fail, LstFile::Skip];

    fn path(self) -> &'static str {
        match self {
            LstFile::Test => CTS_DEFAULT_TEST_LIST,
            LstFile::Fail => "cts_runner/fail.lst",
            LstFile::Skip => "cts_runner/skip.lst",
        }
    }
}

/// A selector pattern from a `.lst` file, precompiled for matching.
struct Pattern {
    /// The literal segments between `*` wildcards. `a*b` gives `["a", "b"]`,
    /// and a pattern with no wildcard gives a single segment.
    segments: Vec<String>,
    /// [`segments`] rewritten as described on [`descending_variant`], or `None`
    /// if the pattern has no trailing path-level wildcard.
    ///
    /// [`segments`]: Pattern::segments
    descending: Option<Vec<String>>,
    source: LstFile,
    has_fails_if: bool,
}

impl Pattern {
    fn new(line: LstLine, source: LstFile) -> Self {
        let segments = line
            .selector
            .split('*')
            .map(str::to_string)
            .collect::<Vec<_>>();
        Self {
            descending: descending_variant(&segments),
            segments,
            source,
            has_fails_if: !line.fails_if.is_empty(),
        }
    }

    /// Whether this pattern matches `selector` in its entirety.
    fn matches(&self, selector: &str) -> bool {
        matches_segments(&self.segments, selector)
            || self
                .descending
                .as_deref()
                .is_some_and(|segments| matches_segments(segments, selector))
    }

    /// The part of the pattern before its first wildcard.
    fn prefix(&self) -> &str {
        self.segments.first().expect("never empty")
    }

    /// The prefix of [`Pattern::descending`], if the pattern has one and it
    /// differs from [`Pattern::prefix`].
    fn descending_prefix(&self) -> Option<&str> {
        let prefix = self.descending.as_ref()?.first().expect("never empty");
        (prefix != self.prefix()).then_some(prefix)
    }

    /// Whether `self` covers strictly more selectors than `narrower`, judged by
    /// whether `narrower` starts within the part `self` matches literally.
    fn is_broader_than(&self, narrower: &Pattern) -> bool {
        let broader = self.prefix();
        let narrower = narrower.prefix();
        if narrower.starts_with(broader) && narrower.len() > broader.len() {
            return true;
        }

        // `a,*` covers everything below `a`, so it is broader than `a:*` even
        // though the two have prefixes of the same length.
        self.descending_prefix()
            .is_some_and(|broader| narrower.starts_with(broader))
    }
}

/// Whether the segments of a pattern match `selector` in its entirety.
fn matches_segments(segments: &[String], selector: &str) -> bool {
    let (first, rest_segments) = segments.split_first().expect("never empty");
    let Some(mut rest) = selector.strip_prefix(first.as_str()) else {
        return false;
    };

    let Some((last, middles)) = rest_segments.split_last() else {
        // No wildcards, so the pattern had to match exactly.
        return rest.is_empty();
    };

    // Matching each middle segment as early as possible leaves the most
    // room for the ones after it, so this never needs to backtrack.
    for middle in middles {
        match rest.find(middle.as_str()) {
            Some(at) => rest = &rest[at + middle.len()..],
            None => return false,
        }
    }

    rest.len() >= last.len() && rest.ends_with(last.as_str())
}

/// The variant of `segments` that descends past a level boundary.
///
/// A CTS query is `suite:file,path:test,path:case=params`, so a trailing `,*`
/// is a wildcard over a `,`-separated path, and covers not just longer paths at
/// that level but everything below the path it names, which continues after a
/// `:`. So `suite:file:test,*` must match `suite:file:test:case=1` as well as
/// `suite:file:test,subtest:case=1`. The returned segments are `segments` with
/// that `,` rewritten as `:`, matching the former; `None` if the pattern has no
/// trailing path-level wildcard.
fn descending_variant(segments: &[String]) -> Option<Vec<String>> {
    let (last, init) = segments.split_last()?;
    if !last.is_empty() {
        return None;
    }
    let stem = init.split_last()?.0.strip_suffix(',')?;

    let mut variant = segments.to_vec();
    let penultimate = variant.len() - 2;
    variant[penultimate] = format!("{stem}:");
    Some(variant)
}

pub fn verify_selectors(
    shell: Shell,
    mut args: Arguments,
    stray_positional: Option<OsString>,
) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");
    let release = args.contains("--release");

    let selectors = stray_positional
        .into_iter()
        .chain(args.finish())
        .map(|s| {
            s.into_string()
                .map_err(|s| anyhow::anyhow!("Selector is not valid UTF-8: {}", s.display()))
        })
        .collect::<anyhow::Result<Vec<String>>>()?;
    let selectors = if selectors.is_empty() {
        DEFAULT_SELECTORS.map(String::from).to_vec()
    } else {
        selectors
    };

    // Load the patterns before checking out the CTS, both to fail fast on a
    // missing `test.lst` and because the checkout moves the shell's cwd.
    let patterns = load_patterns(&shell)?;

    let wgpu_cargo_toml = ensure_cts_checkout(&shell, skip_checkout)?;
    let (bin, env_vars) = build_cts_runner(&shell, &wgpu_cargo_toml, release, false)?;

    let mut all_selectors = Vec::new();
    for selector in &selectors {
        log::info!("Listing CTS selectors for '{selector}'");
        let listed = shell
            .cmd(&bin)
            .envs(env_vars.clone())
            .args(CTS_BIN_ARGS)
            .args(["--list", selector])
            .read()
            .with_context(|| format!("Failed to list CTS selectors for '{selector}'"))?;

        let listed = listed
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty())
            .map(String::from)
            .collect::<Vec<_>>();
        log::info!("  {} selectors", listed.len());
        all_selectors.extend(listed);
    }

    report(&all_selectors, &patterns)
}

/// Read every `.lst` file. `test.lst` must exist; the other two are optional.
fn load_patterns(shell: &Shell) -> anyhow::Result<Vec<Pattern>> {
    let mut patterns = Vec::new();

    for file in LstFile::ALL {
        if !shell.path_exists(file.path()) {
            if file == LstFile::Test {
                bail!("`{}` not found", file.path());
            }
            continue;
        }

        let contents = shell
            .read_file(file.path())
            .with_context(|| format!("Failed to read `{}`", file.path()))?;
        patterns.extend(
            contents
                .lines()
                .filter_map(parse_lst_line)
                .map(|line| Pattern::new(line, file)),
        );
    }

    Ok(patterns)
}

/// Classify every selector, then print the report. Fails if any selector is
/// unmatched or duplicated.
fn report(all_selectors: &[String], patterns: &[Pattern]) -> anyhow::Result<()> {
    let mut passed = BTreeSet::new();
    let mut unmatched = BTreeSet::new();
    let mut duplicated = BTreeSet::new();

    for selector in all_selectors {
        let matching = patterns
            .iter()
            .filter(|pattern| pattern.matches(selector))
            .collect::<Vec<_>>();

        if is_accounted_for(&matching) {
            passed.insert(selector.clone());
        } else if matching.is_empty() {
            unmatched.insert(selector.clone());
        } else {
            duplicated.insert(selector.clone());
        }
    }

    let (red, green, reset) = if std::io::stdout().is_terminal() {
        ("\x1b[0;31m", "\x1b[0;32m", "\x1b[0m")
    } else {
        ("", "", "")
    };

    for (selectors, heading) in [
        (
            &unmatched,
            "Unmatched selectors (not found in any lst file)",
        ),
        (
            &duplicated,
            "Duplicated selectors (found in multiple lst files)",
        ),
    ] {
        if selectors.is_empty() {
            continue;
        }
        println!("{red}{heading}:{reset}");
        for (pattern, matches) in find_minimal_wildcards(selectors, &passed) {
            println!("  {pattern}  ({} selectors)", matches.len());
        }
        println!();
    }

    let failed = unmatched.len() + duplicated.len();

    println!("========================================");
    println!("Summary:");
    println!("  Total selectors:  {}", all_selectors.len());
    println!("  {green}Passed:{reset}           {}", passed.len());
    println!("  {red}Failed:{reset}           {failed}");
    if !unmatched.is_empty() {
        println!("    Unmatched:      {}", unmatched.len());
    }
    if !duplicated.is_empty() {
        println!("    Duplicated:     {}", duplicated.len());
    }
    println!("========================================");

    if failed > 0 {
        bail!("{failed} CTS selector(s) are unmatched or duplicated");
    }

    Ok(())
}

/// Whether the patterns matching a selector give it exactly one unambiguous
/// expectation.
fn is_accounted_for(matching: &[&Pattern]) -> bool {
    if matching.len() == 1 {
        return true;
    }
    if matching.len() != 2 {
        return false;
    }

    // A selector may appear in `test.lst` and in one of the overlay files, so
    // long as the two say something consistent.
    let Some(test) = only_from(matching, LstFile::Test) else {
        return false;
    };
    let Some(overlay) = only_from(matching, LstFile::Fail).or(only_from(matching, LstFile::Skip))
    else {
        return false;
    };

    // The `test.lst` line already says the test fails on some backend, or one
    // line enumerates an exception to the other: either passes are enumerated
    // in `test.lst` under a broader failing/skipped pattern, or a subset of a
    // broader passing pattern is failing or skipped.
    test.has_fails_if || overlay.is_broader_than(test) || test.is_broader_than(overlay)
}

/// The single matching pattern from `file`, or `None` if there isn't exactly one.
fn only_from<'a>(matching: &[&'a Pattern], file: LstFile) -> Option<&'a Pattern> {
    let mut from_file = matching.iter().filter(|pattern| pattern.source == file);
    let first = from_file.next()?;
    from_file.next().is_none().then_some(*first)
}

/// Candidate wildcard prefixes for a selector, most specific first.
///
/// CTS selectors look like
/// `webgpu:shader,execution,expression,call,builtin,abs:*`, so the boundaries
/// worth cutting at are the `:` separators and the `,` separators within the
/// second segment.
fn generate_prefixes(selector: &str) -> Vec<String> {
    let parts = selector.split(':').collect::<Vec<_>>();
    let mut prefixes = Vec::new();

    for i in (1..parts.len()).rev() {
        prefixes.push(format!("{}:", parts[..i].join(":")));
    }

    if let Some(second) = parts.get(1) {
        let subparts = second.split(',').collect::<Vec<_>>();
        for i in (1..subparts.len()).rev() {
            let prefix = format!("{}:{}", parts[0], subparts[..i].join(","));
            prefixes.push(format!("{prefix},"));
            prefixes.push(format!("{prefix}:"));
        }
    }

    prefixes
}

/// Every prefix that any of `selectors` starts with, at the boundaries
/// [`generate_prefixes`] cuts at. Used to reject a candidate wildcard that
/// would swallow a selector we don't want to cover.
fn build_prefix_set(selectors: &BTreeSet<String>) -> BTreeSet<String> {
    let mut prefixes = BTreeSet::new();

    for selector in selectors {
        let parts = selector.split(':').collect::<Vec<_>>();
        for i in 1..=parts.len() {
            let joined = parts[..i].join(":");
            prefixes.insert(format!("{joined}:"));
            prefixes.insert(joined);
        }

        if let Some(second) = parts.get(1) {
            let subparts = second.split(',').collect::<Vec<_>>();
            for i in 1..=subparts.len() {
                let prefix = format!("{}:{}", parts[0], subparts[..i].join(","));
                prefixes.insert(format!("{prefix},"));
                prefixes.insert(format!("{prefix}:"));
                prefixes.insert(prefix);
            }
        }
    }

    prefixes
}

/// Greedily cover `failed` with as few wildcard patterns as possible, without
/// any of them matching a selector in `passed`.
///
/// Returns each pattern with the selectors it covers, most-covering first.
fn find_minimal_wildcards(
    failed: &BTreeSet<String>,
    passed: &BTreeSet<String>,
) -> Vec<(String, Vec<String>)> {
    if failed.is_empty() {
        return Vec::new();
    }

    let passed_prefixes = build_prefix_set(passed);

    let mut candidates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for selector in failed {
        for prefix in generate_prefixes(selector) {
            if passed_prefixes.contains(&prefix) {
                continue;
            }
            candidates
                .entry(prefix)
                .or_default()
                .insert(selector.clone());
        }
    }

    // Considering the widest candidates first makes the greedy search below
    // converge sooner. Between candidates covering the same selectors, prefer
    // the most specific, so that covering one stray selector doesn't suggest
    // a wildcard broad enough to swallow half the CTS. The final comparison
    // just keeps the output stable from run to run.
    let mut by_coverage = candidates.keys().cloned().collect::<Vec<_>>();
    by_coverage.sort_by(|a, b| {
        candidates[b]
            .len()
            .cmp(&candidates[a].len())
            .then_with(|| b.len().cmp(&a.len()))
            .then_with(|| a.cmp(b))
    });

    let mut uncovered = failed.clone();
    let mut result = Vec::new();

    while !uncovered.is_empty() {
        let mut best_prefix = None;
        let mut best_matches = Vec::new();

        for prefix in &by_coverage {
            let Some(covers) = candidates.get(prefix) else {
                continue;
            };
            let matches = covers.intersection(&uncovered).cloned().collect::<Vec<_>>();
            if matches.len() > best_matches.len() {
                best_matches = matches;
                best_prefix = Some(prefix.clone());
            }
        }

        let Some(best_prefix) = best_prefix else {
            // Nothing left that we're allowed to generalize; name the
            // stragglers individually.
            result.extend(uncovered.iter().map(|s| (s.clone(), vec![s.clone()])));
            break;
        };

        for selector in &best_matches {
            uncovered.remove(selector);
        }
        candidates.remove(&best_prefix);
        result.push((format!("{best_prefix}*"), best_matches));
    }

    // A stable sort, so patterns covering the same number of selectors stay in
    // the order the search found them.
    result.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(selector: &str) -> Pattern {
        pattern_from(selector, LstFile::Test, &[])
    }

    fn pattern_from(selector: &str, source: LstFile, fails_if: &[&str]) -> Pattern {
        Pattern::new(
            LstLine {
                selector: selector.to_string(),
                fails_if: fails_if.iter().map(|s| s.to_string()).collect(),
            },
            source,
        )
    }

    #[test]
    fn test_matches_without_wildcards() {
        let p = pattern("webgpu:api,validation,buffer,mapping:mapAsync,usage:");
        assert!(p.matches("webgpu:api,validation,buffer,mapping:mapAsync,usage:"));
        assert!(!p.matches("webgpu:api,validation,buffer,mapping:mapAsync,usage:extra"));
        assert!(!p.matches("webgpu:api,validation,buffer,mapping:mapAsync,usag"));
        assert!(!p.matches(""));
    }

    #[test]
    fn test_matches_trailing_wildcard() {
        let p = pattern("webgpu:api,validation,*");
        assert!(p.matches("webgpu:api,validation,"));
        assert!(p.matches("webgpu:api,validation,buffer,mapping:mapAsync,usage:"));
        assert!(!p.matches("webgpu:api,validation"));
        assert!(!p.matches("webgpu:api,operation,buffer:"));
    }

    #[test]
    fn test_matches_leading_and_middle_wildcards() {
        let p = pattern("*mapping:*");
        assert!(p.matches("webgpu:api,validation,buffer,mapping:mapAsync"));
        assert!(p.matches("mapping:"));
        assert!(!p.matches("webgpu:api,validation,buffer,mapping"));

        let p = pattern("webgpu:*:mapAsync,*");
        assert!(p.matches("webgpu:api,validation,buffer,mapping:mapAsync,usage:"));
        assert!(!p.matches("webgpu:api,validation,buffer,mapping:unmapped,usage:"));
    }

    #[test]
    fn test_matches_path_wildcard_descends_a_level() {
        // Regression test: a trailing `,*` covers everything below the path it
        // names, including the case level, which starts after a `:`.
        let p = pattern("webgpu:api,validation,createTexture:texture_usage,*");
        assert!(p.matches("webgpu:api,validation,createTexture:texture_usage:dimension=\"2d\""));
        assert!(p.matches("webgpu:api,validation,createTexture:texture_usage,subtest:"));
        // The `:` has to fall on the level boundary the wildcard sits at.
        assert!(!p
            .matches("webgpu:api,validation,createTexture:texture_usage_extra:format=\"r8unorm\""));
        assert!(!p.matches("webgpu:api,validation,createTexture:texture_usageFoo"));
        assert!(!p.matches("webgpu:api,validation,createTexture:texture_usag:"));

        // The same at the file-path level.
        let p = pattern("webgpu:api,validation,buffer,create,*");
        assert!(p.matches("webgpu:api,validation,buffer,create:size:"));
        assert!(p.matches("webgpu:api,validation,buffer,create,more:size:"));
        assert!(!p.matches("webgpu:api,validation,buffer,createFoo:size:"));

        // A `:*` wildcard stays within the one test it names.
        let p = pattern("webgpu:api,validation,createTexture:texture_usage:*");
        assert!(p.matches("webgpu:api,validation,createTexture:texture_usage:dimension=\"2d\""));
        assert!(!p.matches("webgpu:api,validation,createTexture:texture_usage,subtest:"));
    }

    #[test]
    fn test_matches_case_param_wildcard() {
        let p = pattern("webgpu:api,validation,createTexture:texture_usage:isAsync=false;*");
        assert!(p.matches(
            "webgpu:api,validation,createTexture:texture_usage:isAsync=false;format=\"r8unorm\""
        ));
        assert!(!p.matches(
            "webgpu:api,validation,createTexture:texture_usage:isAsync=true;format=\"r8unorm\""
        ));
        // A case-param wildcard is not a path wildcard, so it doesn't descend.
        assert!(!p.matches("webgpu:api,validation,createTexture:texture_usage:isAsync=false:x"));
    }

    #[test]
    fn test_matches_degenerate_wildcards() {
        assert!(pattern("*").matches(""));
        assert!(pattern("*").matches("anything at all"));
        // Consecutive wildcards are equivalent to one.
        assert!(pattern("a**b").matches("ab"));
        assert!(pattern("a**b").matches("axxxb"));
        assert!(!pattern("a**b").matches("a"));
        // The prefix and suffix must not overlap.
        assert!(!pattern("ab*bc").matches("abc"));
        assert!(pattern("ab*bc").matches("abbc"));
    }

    #[test]
    fn test_is_broader_than() {
        let broad = pattern("webgpu:api,validation,*");
        let narrow = pattern("webgpu:api,validation,buffer,*");
        assert!(broad.is_broader_than(&narrow));
        assert!(!narrow.is_broader_than(&broad));
        // A pattern is not broader than itself.
        assert!(!broad.is_broader_than(&pattern("webgpu:api,validation,*")));
        // Siblings are unrelated.
        assert!(!narrow.is_broader_than(&pattern("webgpu:api,validation,texture,*")));
    }

    #[test]
    fn test_is_broader_than_across_a_level_boundary() {
        let broad = pattern("webgpu:api,validation,createTexture:texture_usage,*");
        for narrow in [
            pattern("webgpu:api,validation,createTexture:texture_usage:*"),
            pattern("webgpu:api,validation,createTexture:texture_usage:dimension=\"2d\""),
            pattern("webgpu:api,validation,createTexture:texture_usage,subtest:*"),
        ] {
            assert!(broad.is_broader_than(&narrow));
            assert!(!narrow.is_broader_than(&broad));
        }
        // A pattern is still not broader than itself.
        assert!(!broad.is_broader_than(&pattern(
            "webgpu:api,validation,createTexture:texture_usage,*"
        )));
        // Nor is it broader than a sibling reached over the same boundary.
        assert!(!broad.is_broader_than(&pattern(
            "webgpu:api,validation,createTexture:texture_usage_extra:*"
        )));
    }

    #[test]
    fn test_is_accounted_for() {
        let test = pattern_from("webgpu:a,*", LstFile::Test, &[]);
        let test_narrow = pattern_from("webgpu:a,b,*", LstFile::Test, &[]);
        let test_fails_if = pattern_from("webgpu:a,*", LstFile::Test, &["vulkan"]);
        let fail = pattern_from("webgpu:a,*", LstFile::Fail, &[]);
        let fail_narrow = pattern_from("webgpu:a,b,*", LstFile::Fail, &[]);
        let skip_narrow = pattern_from("webgpu:a,b,*", LstFile::Skip, &[]);

        // Exactly one match is always fine; none or three is never fine.
        assert!(is_accounted_for(&[&test]));
        assert!(!is_accounted_for(&[]));
        assert!(!is_accounted_for(&[&test, &fail, &skip_narrow]));

        // The same pattern in test.lst and fail.lst is ambiguous, unless the
        // test.lst line already says it fails somewhere.
        assert!(!is_accounted_for(&[&test, &fail]));
        assert!(is_accounted_for(&[&test_fails_if, &fail]));

        // One line enumerating an exception to the other is fine, in either
        // direction, and for both overlay files.
        assert!(is_accounted_for(&[&test, &fail_narrow]));
        assert!(is_accounted_for(&[&test, &skip_narrow]));
        assert!(is_accounted_for(&[&test_narrow, &fail]));

        // Two matches that don't involve test.lst are ambiguous.
        assert!(!is_accounted_for(&[&fail, &skip_narrow]));
        // As are two matches within test.lst itself.
        assert!(!is_accounted_for(&[&test, &test_narrow]));
    }

    #[test]
    fn test_generate_prefixes() {
        assert_eq!(
            generate_prefixes("webgpu:api,validation,buffer:mapping"),
            [
                "webgpu:api,validation,buffer:",
                "webgpu:",
                "webgpu:api,validation,",
                "webgpu:api,validation:",
                "webgpu:api,",
                "webgpu:api:",
            ]
        );
        assert_eq!(generate_prefixes("unittests"), Vec::<String>::new());
    }

    #[test]
    fn test_find_minimal_wildcards_covers_everything() {
        let failed = ["webgpu:a,b:x", "webgpu:a,b:y", "webgpu:a,c:z"]
            .map(String::from)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let wildcards = find_minimal_wildcards(&failed, &BTreeSet::new());

        // With nothing passing to avoid, one wildcard should cover all three.
        assert_eq!(wildcards.len(), 1);
        assert_eq!(wildcards[0].1.len(), 3);
        let covering = pattern(&wildcards[0].0);
        assert!(failed.iter().all(|s| covering.matches(s)));
    }

    #[test]
    fn test_find_minimal_wildcards_avoids_passing_selectors() {
        let failed = ["webgpu:a,b:x", "webgpu:a,b:y"]
            .map(String::from)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let passed = ["webgpu:a,c:z"]
            .map(String::from)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let wildcards = find_minimal_wildcards(&failed, &passed);

        for (wildcard, _) in &wildcards {
            let wildcard = pattern(wildcard);
            assert!(
                !passed.iter().any(|s| wildcard.matches(s)),
                "`{}` swallows a passing selector",
                wildcard.segments.join("*"),
            );
        }
        // Every failing selector is still covered.
        assert!(failed
            .iter()
            .all(|s| wildcards.iter().any(|(w, _)| pattern(w).matches(s))));
    }
}
