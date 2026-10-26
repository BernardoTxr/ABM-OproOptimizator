from workflows.validation.smoke import run_smoke


def test_offline_smoke_exercises_simulator_and_both_opro_conditions() -> None:
    result = run_smoke()

    assert result["status"] == "ready"
    assert result["simulator"]["scenario_count"] == 1
    assert result["simulator"]["steps"] == 5
    assert set(result["opro"]["conditions"]) == {"plain", "heuristic"}
    assert result["opro"]["conditions"]["plain"]["heuristic_condition"] is False
    assert result["opro"]["conditions"]["heuristic"]["heuristic_condition"] is True
    assert result["opro"]["conditions"]["plain"]["history_observations"] == 49
    assert result["opro"]["conditions"]["heuristic"]["history_observations"] == 49
    assert result["opro"]["conditions"]["plain"]["remaining_proposal_budget"] == 1
    assert result["opro"]["conditions"]["heuristic"][
        "heuristic_registry_version"
    ] == "component_mechanisms_smm_v1"
    assert result["opro"]["conditions"]["plain"]["cost_status"] == (
        "true_zero_offline"
    )
    assert result["opro"]["network_calls"] == 0
    assert result["opro"]["optimizer_slots_consumed"] == 0
    assert result["opro"]["private_data_accessed"] is False
