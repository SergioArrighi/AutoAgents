use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

fn load_json(path: &Path) -> Value {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("invalid json {}: {err}", path.display()))
}

fn as_object<'a>(value: &'a Value, path: &str) -> &'a serde_json::Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("expected object at {path}"))
}

fn as_str<'a>(value: &'a Value, path: &str) -> &'a str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("expected string at {path}"))
}

fn eval_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("eval")
}

#[test]
fn eval_contract_offline_scoring() {
    let eval_dir = eval_dir();
    let taxonomy_path = eval_dir.join("query_taxonomy.json");
    let contract_path = eval_dir.join("goldens/rust_copilot_metrics_fixture.contract.json");
    let observed_path = std::env::var("RUST_COPILOT_EVAL_OBSERVED_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| eval_dir.join("observed/rust_copilot_metrics_fixture.sample.json"));

    let taxonomy = load_json(&taxonomy_path);
    let contract = load_json(&contract_path);
    let observed = load_json(&observed_path);

    let taxonomy_queries = taxonomy
        .get("queries")
        .and_then(Value::as_array)
        .expect("query_taxonomy.json: queries must be an array");
    assert!(
        !taxonomy_queries.is_empty(),
        "query_taxonomy.json: queries must not be empty"
    );

    let mut query_ids = HashSet::new();
    for query in taxonomy_queries {
        let id = as_str(
            query
                .get("id")
                .unwrap_or_else(|| panic!("missing query id in {}", taxonomy_path.display())),
            "queries[].id",
        );
        assert!(
            query_ids.insert(id.to_string()),
            "duplicate query id in taxonomy: {id}"
        );
    }

    let gates = as_object(
        contract
            .get("gates")
            .expect("contract: missing gates object"),
        "contract.gates",
    );

    let status = observed
        .get("index_status")
        .and_then(|v| v.get("status"))
        .expect("observed: missing index_status.status");
    let status_obj = as_object(status, "observed.index_status.status");

    let extraction_metrics = as_object(
        status_obj
            .get("extraction_metrics")
            .expect("observed: missing extraction_metrics"),
        "observed.index_status.status.extraction_metrics",
    );
    let expected_extraction = as_object(
        gates
            .get("extraction_metrics_exact")
            .expect("contract: missing extraction_metrics_exact"),
        "contract.gates.extraction_metrics_exact",
    );

    let expected_index_status = as_object(
        gates
            .get("index_status_exact")
            .expect("contract: missing index_status_exact"),
        "contract.gates.index_status_exact",
    );
    let expected_qdrant = as_object(
        gates
            .get("qdrant_points_exact")
            .expect("contract: missing qdrant_points_exact"),
        "contract.gates.qdrant_points_exact",
    );
    let observed_qdrant = as_object(
        observed
            .get("qdrant_points")
            .expect("observed: missing qdrant_points"),
        "observed.qdrant_points",
    );

    let expected_retrieval = as_object(
        gates
            .get("retrieval_assertions")
            .expect("contract: missing retrieval_assertions"),
        "contract.gates.retrieval_assertions",
    );
    let observed_retrieval = as_object(
        observed
            .get("retrieval_assertions")
            .expect("observed: missing retrieval_assertions"),
        "observed.retrieval_assertions",
    );

    let mut checks_total = 0u64;
    let mut checks_passed = 0u64;
    let mut failures = Vec::<String>::new();

    for (key, expected_value) in expected_index_status {
        checks_total += 1;
        let observed_value = status_obj
            .get(key)
            .unwrap_or_else(|| panic!("observed index status missing key: {key}"));
        if observed_value == expected_value {
            checks_passed += 1;
        } else {
            failures.push(format!(
                "index_status_exact.{key}: expected {expected_value}, got {observed_value}"
            ));
        }
    }

    for (key, expected_value) in expected_qdrant {
        checks_total += 1;
        let observed_value = observed_qdrant
            .get(key)
            .unwrap_or_else(|| panic!("observed qdrant_points missing key: {key}"));
        if observed_value == expected_value {
            checks_passed += 1;
        } else {
            failures.push(format!(
                "qdrant_points_exact.{key}: expected {expected_value}, got {observed_value}"
            ));
        }
    }

    for (key, expected_value) in expected_extraction {
        checks_total += 1;
        let observed_value = extraction_metrics
            .get(key)
            .unwrap_or_else(|| panic!("observed extraction_metrics missing key: {key}"));
        if observed_value == expected_value {
            checks_passed += 1;
        } else {
            failures.push(format!(
                "extraction_metrics_exact.{key}: expected {expected_value}, got {observed_value}"
            ));
        }
    }

    for (query_id, expected_rule) in expected_retrieval {
        assert!(
            query_ids.contains(query_id),
            "retrieval_assertions references unknown taxonomy id: {query_id}"
        );
        let expected_rule = as_object(expected_rule, "contract.gates.retrieval_assertions.*");
        let observed_rule = as_object(
            observed_retrieval
                .get(query_id)
                .unwrap_or_else(|| panic!("observed retrieval_assertions missing: {query_id}")),
            "observed.retrieval_assertions.*",
        );
        for (field, expected_value) in expected_rule {
            checks_total += 1;
            let observed_field_name = field.strip_prefix("expected_").unwrap_or_else(|| {
                panic!("expected retrieval key must start with 'expected_': {field}")
            });
            let observed_value = observed_rule.get(observed_field_name).unwrap_or_else(|| {
                panic!(
                    "observed retrieval assertion missing field '{observed_field_name}' for '{query_id}'"
                )
            });
            if observed_value == expected_value {
                checks_passed += 1;
            } else {
                failures.push(format!(
                    "retrieval_assertions.{query_id}.{field}: expected {expected_value}, got {observed_value}"
                ));
            }
        }
    }

    let score = if checks_total == 0 {
        0.0
    } else {
        (checks_passed as f64 / checks_total as f64) * 10.0
    };
    let min_score = gates
        .get("min_contract_score")
        .and_then(Value::as_f64)
        .expect("contract: gates.min_contract_score must be f64");

    println!(
        "Eval contract score: {:.2}/10 (passed {checks_passed}/{checks_total})",
        score
    );
    println!("Observed snapshot: {}", observed_path.display());

    assert!(
        failures.is_empty(),
        "eval contract mismatches:\n{}",
        failures.join("\n")
    );
    assert!(
        score >= min_score,
        "eval contract score {:.2}/10 is below target {:.2}/10",
        score,
        min_score
    );
}

#[test]
fn eval_contract_files_exist() {
    let eval_dir = eval_dir();
    for rel in [
        "query_taxonomy.json",
        "goldens/rust_copilot_metrics_fixture.contract.json",
        "observed/rust_copilot_metrics_fixture.sample.json",
    ] {
        let path = eval_dir.join(rel);
        assert!(path.exists(), "missing eval file: {}", path.display());
    }
}
