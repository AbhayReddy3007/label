"""
Serious Safety Profile Scoring Logic

Implements a deterministic, multi-step scoring algorithm for clinical trial serious safety profiles.

Logic overview:
1. Trials are banded by delta SAE (S1-S4) to determine the dominant risk pattern.
2. Base score is set by the dominant band (S4 forces score 2; otherwise, S1–S3 by high-tier count).
3. Pattern adjustments:
    - Death/irreversible harm (severity 4) in ≥2 trials with positive delta SAE forces score to 1.
    - Life-threatening events (severity 3) in S4 reduce score by 1.
    - Repeated life-threatening or organ-specific events across ≥2 trials reduce score by 1 each.
4. Expectedness adjustment:
    - Clinically meaningful unexpected or borderline (expected but rare) events reduce score by 1 each.
5. Regulatory cap: If regulatory consequence applies, score is capped.
6. Post-marketing adjustment: New/stronger post-marketing risks reduce score by 1; post-marketing cap may further limit score.
7. Final score is clamped to [1, 5].

All adjustments and key drivers are recorded for traceability."""

from __future__ import annotations

from .adjustments import get_post_marketing_cap, get_regulatory_cap
from .constants import BAND_TO_SCORE, DEFAULT_SCORE_NO_DELTA_SAE, DEFAULT_SCORE_NO_SAE
from .parsing import derive_trial_delta, severity_level, calculate_presence_rate, is_clinically_meaningful, parse_float
from .trial_banding import build_trial_banding


def score_serious_safety_profile(
    sae_aggregation: dict,
    sae_event_categorization: list,
    regulatory_impact: dict,
    post_marketing_safety: dict,
    is_marketed: bool = False
) -> dict:
    """Score serious safety profile using deterministic multi-step business logic."""
    # Shared accumulators used for explanation output and final decision trace.
    reasoning: dict = {}
    adjustments_applied: list[str] = []
    key_drivers: list[str] = []
    trials = list(sae_aggregation.get("trials") or [])

    # Step 1: Build trial-level delta bands (S1-S4).
    trials, band_counts, high_tier_band_counts, has_s4 = build_trial_banding(
        trials=trials,
        derive_trial_delta=derive_trial_delta,
    )

    reasoning["step1_trial_banding"] = {
        "band_counts": band_counts,
        "high_tier_band_counts": high_tier_band_counts,
    }

    # Step 2: Determine base score from band pattern (S4 forces score 2).
    all_delta_null = bool(trials) and all(parse_float(trial.get("Delta SAE (%)")) is None for trial in trials)
    all_sae_rates_zero = bool(trials) and all(
        (parse_float(trial.get("SAE Rate Drug (%)")) in (0.0, None)) and
        (parse_float(trial.get("SAE Rate Control (%)")) in (0.0, None))
        for trial in trials
    )

    if all_delta_null or all_sae_rates_zero:
        base_score = DEFAULT_SCORE_NO_DELTA_SAE
        dominant_band = "N/A"
        base_note = "No usable SAE rate data found for trials; applying default base score"
    elif has_s4:
        base_score = 2
        dominant_band = "S4"
        base_note = "At least one trial with delta SAE > 2.0, base score forced to 2"
    else:
        eligible = ["S1", "S2", "S3"]
        # Checks which category among the above has the highes number of "High" tier trials
        # If two categories have same count, the one with lower score is picked
        dominant_band = max(eligible, key=lambda b: (high_tier_band_counts[b], -BAND_TO_SCORE[b]))
        base_score = BAND_TO_SCORE[dominant_band]
        base_note = f"Dominant band: {dominant_band}, has the {high_tier_band_counts[dominant_band]} number of high weightage trials"

    score = base_score
    adjustments_applied.append("base_score_assigned")
    key_drivers.append(base_note)
    reasoning["step2_base_score"] = {
        "dominant_band": dominant_band,
        "score": base_score,
        "note": base_note,
    }

    # Step 3: Pattern adjustment - severity checks and repeated SAE categories.
    all_events = sae_event_categorization or []
    
    # First check for death or irreversible harm (severity level 4)
    death_irreversible_events = [ev for ev in all_events if severity_level(ev.get("severity_category", "")) == 4]
    
    # Collect trial_ids from category 4 events where delta_sae > 0
    drug_related_death_trial_ids_with_positive_delta = set()
    for ev in death_irreversible_events:
        studies = ev.get("list_of_studies", []) or []
        if studies:
            for trial_id in studies:
                # Find the trial in trials list by Trial ID
                for trial in trials:
                    if str(trial.get("Trial ID", "")).strip() == str(trial_id).strip():
                        delta_sae = derive_trial_delta(trial)
                        drug_related_deaths = trial.get('Drug-Related Deaths',"").strip().lower()
                        if delta_sae is not None and delta_sae > 0 and drug_related_deaths == 'yes':
                            drug_related_death_trial_ids_with_positive_delta.add(trial_id)
                        break
    
    pattern_adj = 0
    pattern_note = "No qualifying pattern"
    life_threatening_events = []
    repeated_events = []
    repeated_organ = []
    repeated_life = []

    if not all_events:
        pattern_adj = 0
        score = DEFAULT_SCORE_NO_SAE
        pattern_note = "No SAE events found for these trials; applying default score"
        adjustments_applied.append("no_sae_events_default_score_applied")
        key_drivers.append(pattern_note)

    elif len(drug_related_death_trial_ids_with_positive_delta) >= 2:
        # Death or irreversible harm found - set base score to 1
        pattern_adj = -(base_score - 1)
        score = 1
        pattern_note = f"({len(death_irreversible_events)}) number death or fatal events found, \
            with {len(drug_related_death_trial_ids_with_positive_delta)} trials having positive delta SAE and drug related deaths, base score set to 1"
        key_drivers.append(f"Death or fatal events with positive delta SAE and drug related deaths in {len(drug_related_death_trial_ids_with_positive_delta)} trials forces score to 1")

    else:

        # To define repeated events, threshold is 5% trials for marketed drugs and 2 for approved drugs
        repeat_event_theshold = 2 if is_marketed == False else round(len(trials)*0.05)

        # Check for repeated events in life-threatening and organ categories
        repeated_events = [
            ev for ev in all_events
            if int(ev.get("number_of_studies") or 0) >= repeat_event_theshold
        ]

        repeated_organ = [ev for ev in repeated_events if severity_level(ev.get("severity_category", "")) == 2]
        repeated_life = [ev for ev in repeated_events if severity_level(ev.get("severity_category", "")) == 3]

        pattern_notes = []

        if has_s4:
            # Check for life-threatening events (severity level == 3)
            life_threatening_events = [ev for ev in all_events if severity_level(ev.get("severity_category", "")) == 3]

            if life_threatening_events:
                # Life-threatening event found - reduce by 1
                pattern_adj = -1
                score = max(1, score + pattern_adj)
                pattern_note = f"Life-threatening severity event found ({len(life_threatening_events)}), \
                    and as S4 category trial exists score is reduced by 1"
                adjustments_applied.append("pattern_life_threatening_minus1")
                key_drivers.append(pattern_note)

            if repeated_life:
                pattern_adj -= 1
                score = max(1, score - 1)
                pattern_notes.append(f"Repeated life-threatening/severe SAE category found across >={repeat_event_theshold} trials")
                adjustments_applied.append("pattern_repeated_life_minus1")

        if repeated_organ:
            pattern_adj -= 1
            score = max(1, score - 1)
            pattern_notes.append(f"Repeated organ-specific SAE category found across >={repeat_event_theshold} trials")
            adjustments_applied.append("pattern_repeated_organ_minus1")

        if pattern_notes:
            pattern_note = "; ".join(pattern_notes)
            for note in pattern_notes:
                key_drivers.append(note)

    reasoning["step3_pattern_adjustment"] = {
        "death_fatal_count": len(death_irreversible_events),
        "drug_related_death_fatal_count": len(drug_related_death_trial_ids_with_positive_delta),
        "repeated_event_count": len(repeated_events),
        "repeated_organ_count": len(repeated_organ),
        "repeated_life_count": len(repeated_life),
        "adjustment": pattern_adj,
        "score_after": score,
        "note": pattern_note,
    }

    # Step 4: Expectedness adjustment on material risks (repeated or severe events).
    min_num_for_repeated_events = round(len(trials) * 0.03)
    material_events = []
    clinically_meaningful_unexpected = []
    clinically_meaningful_borderline = []
    clinically_meaningful_expected = []

    # Calculate total number of studies for presence rate calculation
    total_studies = len(trials) if trials else 1

    expectedness_adj = 0
    expectedness_note = "No material unexpected/borderline safety signal"

    if not (sae_event_categorization or []):
        expectedness_adj = 0
        score = DEFAULT_SCORE_NO_SAE
        expectedness_note = "No data found related to serious adverse events for these trials; applying default score"
        adjustments_applied.append("expectedness_no_sae_events_default_score_applied")
        key_drivers.append(expectedness_note)
    else:
        for ev in (sae_event_categorization or []):
            sev = severity_level(ev.get("severity_category", ""))
            repeated = int(ev.get("number_of_studies") or 0) > min_num_for_repeated_events
            is_material = repeated or sev >= 3
            # Add material_event_flag to the event
            ev["material_event_flag"] = is_material
            if is_material:
                material_events.append(ev)

        # Separate events into clinically meaningful categories
        for ev in material_events:
            expectedness = str(ev.get("expectedness_classification", "")).strip().lower()
            is_meaningful = is_clinically_meaningful(ev, total_studies)

            if is_meaningful:
                if expectedness == "unexpected":
                    clinically_meaningful_unexpected.append(ev)
                elif expectedness == "expected_but_rare":
                    clinically_meaningful_borderline.append(ev)
                else:
                    clinically_meaningful_expected.append(ev)

        expectedness_notes = []

        # Apply adjustment rules based on clinical meaningfulness
        if clinically_meaningful_unexpected:
            expectedness_adj -= 1
            score = max(1, score - 1)
            expectedness_notes.append(f"Total of {len(clinically_meaningful_unexpected)} clinically meaningful unexpected SAE signal detected")
            adjustments_applied.append("expectedness_unexpected_minus1")

        if clinically_meaningful_borderline:
            expectedness_adj -= 1
            score = max(1, score - 1)
            expectedness_notes.append(f"Total of {len(clinically_meaningful_borderline)} clinically meaningful borderline (expected but rare) SAE signal detected")
            adjustments_applied.append("expectedness_borderline_minus1")

        if expectedness_notes:
            expectedness_note = "; ".join(expectedness_notes)
            for note in expectedness_notes:
                key_drivers.append(note)

    reasoning["step4_expectedness_adjustment"] = {
        "material_event_count": len(material_events), 
        "total_studies": total_studies,
        "clinically_meaningful_unexpected_count": len(clinically_meaningful_unexpected),
        "clinically_meaningful_borderline_count": len(clinically_meaningful_borderline),
        "clinically_meaningful_expected_count": len(clinically_meaningful_expected),
        "adjustment": expectedness_adj,
        "score_after": score,
        "note": expectedness_note,
    }

    # Step 5: Apply regulatory cap (if any) to the in-progress score.
    reg_cap, reg_note = get_regulatory_cap(regulatory_impact or {})
    if reg_cap is not None and score > reg_cap:
        score = reg_cap
        adjustments_applied.append("regulatory_cap_applied")
        key_drivers.append(reg_note)

    reasoning["step5_regulatory_adjustment"] = {
        "regulatory_consequence": regulatory_impact.get("regulatory_consequence", ""),
        "consequence_level": regulatory_impact.get("consequence_level"),
        "cap": reg_cap,
        "score_after": score,
        "note": reg_note,
    }

    # Step 6: Apply post-marketing adjustment and combine post-marketing cap with regulatory cap.
    pm = post_marketing_safety or {}
    pm_adj = 0
    pm_note = "No post-marketing adjustment"
    if pm.get("evaluated") and pm.get("new_serious_risks"):
        pm_adj = -1
        score = max(1, score + pm_adj)
        pm_note = "New or stronger post-marketing safety signals identified"
        adjustments_applied.append("post_marketing_minus1")
        key_drivers.append(pm_note)

    pm_cap, pm_cap_note = get_post_marketing_cap(pm)
    if pm_cap is not None:
        pm_note = f"{pm_note}; {pm_cap_note}"

    if pm_cap is not None and score > pm_cap:
        score = pm_cap
        adjustments_applied.append("final_cap_applied")

    reasoning["step6_post_marketing_adjustment"] = {
        "evaluated": pm.get("evaluated", False),
        "adjustment": pm_adj,
        "post_marketing_cap": pm_cap,
        "score_after": score,
        "note": pm_note,
    }

    # Step 7: Final score clamp and output label mapping.
    score = max(1, min(5, score))

    reasoning["step7_final"] = {
        "score": score,
        "adjustments_applied": adjustments_applied,
    }

    return {
        "score": score,
        "reasoning": reasoning,
        "key_drivers": key_drivers,
        "adjustments_applied": adjustments_applied,
        "trials": trials,
        "sae_event_categorization":sae_event_categorization
    }
