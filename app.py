"""Optional Streamlit dashboard for local multimodal session reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
UPLOADS = {
    "attendance": ("Attendance records", "attendance.csv"),
    "scores": ("Activity score timeline", "scores.csv"),
    "suggestions": ("Teaching suggestions", "suggestions.csv"),
    "participants": ("Student session analytics", "participants.csv"),
    "group_timeline": ("Group activity timeline", "group_timeline.csv"),
    "group_summary": ("Group activity summary", "group_summary.csv"),
}


def _load_upload_or_output(uploaded_file, output_name: str) -> pd.DataFrame | None:
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        output_path = OUTPUT_DIRECTORY / output_name
        if output_path.is_file():
            return pd.read_csv(output_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        st.error(f"Could not load {output_name}: {error}")
    return None


def _demo_data() -> dict[str, pd.DataFrame]:
    return {
        "attendance": pd.DataFrame(
            [
                {
                    "tracking_id": 1,
                    "observed_presence_seconds": 48.0,
                    "required_presence_seconds": 45.0,
                    "meets_presence_duration_threshold": True,
                    "observation_count": 96,
                    "presence_bouts": 1,
                    "individual_activity_consistency": None,
                },
                {
                    "tracking_id": 2,
                    "observed_presence_seconds": 31.5,
                    "required_presence_seconds": 45.0,
                    "meets_presence_duration_threshold": False,
                    "observation_count": 63,
                    "presence_bouts": 2,
                    "individual_activity_consistency": None,
                },
            ]
        ),
        "scores": pd.DataFrame(
            [
                {
                    "window_index": index,
                    "start_seconds": index * 30,
                    "end_seconds": (index + 1) * 30,
                    "activity_score": score,
                    "activity_band": band,
                    "confidence": "low",
                    "explanation": (
                        "Synthetic group-level observed-activity example; "
                        "not an engagement measurement."
                    ),
                }
                for index, (score, band) in enumerate(
                    [
                        (72.0, "higher_observed_activity"),
                        (58.0, "moderate_observed_activity"),
                        (32.0, "lower_observed_activity"),
                        (29.0, "lower_observed_activity"),
                        (67.0, "higher_observed_activity"),
                        (74.0, "higher_observed_activity"),
                    ]
                )
            ]
        ),
        "suggestions": pd.DataFrame(
            [
                {
                    "suggestion_type": "sustained_lower_activity_proxy",
                    "priority": "review",
                    "start_seconds": 60,
                    "end_seconds": 120,
                    "observation": "Synthetic example: activity proxy remained lower.",
                    "suggestion": (
                        "Consider a short change of pace or whole-class check-in."
                    ),
                    "evidence": "Synthetic demonstration data only.",
                    "confidence": "low",
                    "human_review_recommended": True,
                },
                {
                    "suggestion_type": "higher_activity_proxy_period",
                    "priority": "information",
                    "start_seconds": 120,
                    "end_seconds": 180,
                    "observation": "Synthetic example: activity proxy was higher.",
                    "suggestion": (
                        "You may wish to note the activity format for reflection."
                    ),
                    "evidence": "Synthetic demonstration data only.",
                    "confidence": "low",
                    "human_review_recommended": True,
                },
            ]
        ),
        "participants": pd.DataFrame(
            [
                {
                    "temporary_tracking_id": 1,
                    "observed_presence_seconds": 48.0,
                    "required_presence_seconds": 45.0,
                    "meets_presence_duration_threshold": True,
                    "observation_count": 96,
                    "presence_bouts": 1,
                    "individual_activity_consistency": None,
                    "individual_activity_timeline_available": False,
                    "reflection_note": (
                        "Synthetic track only; not a verified identity."
                    ),
                }
            ]
        ),
        "group_timeline": pd.DataFrame(
            [
                {
                    "window_index": index,
                    "start_seconds": index * 30,
                    "end_seconds": (index + 1) * 30,
                    "activity_score": score,
                    "activity_band": band,
                    "confidence": "low",
                    "explanation": (
                        "Synthetic group-level activity example; "
                        "not individual engagement."
                    ),
                }
                for index, (score, band) in enumerate(
                    [
                        (72.0, "higher_observed_activity"),
                        (58.0, "moderate_observed_activity"),
                        (32.0, "lower_observed_activity"),
                        (29.0, "lower_observed_activity"),
                        (67.0, "higher_observed_activity"),
                        (74.0, "higher_observed_activity"),
                    ]
                )
            ]
        ),
        "group_summary": pd.DataFrame(
            [
                {
                    "scored_window_count": 6,
                    "unscored_window_count": 0,
                    "lower_activity_window_count": 2,
                    "moderate_activity_window_count": 1,
                    "higher_activity_window_count": 3,
                    "mean_activity_score": 55.3,
                    "score_standard_deviation": 18.1,
                    "summary": "Synthetic session example only.",
                }
            ]
        ),
    }


def _is_usable(data: pd.DataFrame | None, columns: set[str]) -> bool:
    if data is None:
        return False
    missing = columns.difference(data.columns)
    if missing:
        st.warning(
            "This CSV is missing columns required for its visualization: "
            + ", ".join(sorted(missing))
        )
        return False
    return True


def _render_activity_timeline(
    scores: pd.DataFrame | None,
    *,
    chart_key: str,
) -> None:
    st.subheader("Group activity timeline")
    if not _is_usable(
        scores,
        {"start_seconds", "end_seconds", "activity_score", "activity_band"},
    ):
        st.info("Upload an activity-score CSV to see the session timeline.")
        return

    timeline = scores.copy()
    timeline["activity_score"] = pd.to_numeric(
        timeline["activity_score"], errors="coerce"
    )
    timeline["start_seconds"] = pd.to_numeric(
        timeline["start_seconds"], errors="coerce"
    )
    timeline = timeline.dropna(subset=["start_seconds", "activity_score"])
    if timeline.empty:
        st.info("No scored activity windows are available to chart.")
        return

    fig = px.line(
        timeline,
        x="start_seconds",
        y="activity_score",
        color="activity_band",
        markers=True,
        hover_data=["end_seconds", "confidence"]
        if "confidence" in timeline.columns
        else ["end_seconds"],
        labels={
            "start_seconds": "Session time (seconds)",
            "activity_score": "Observed activity proxy (0-100)",
            "activity_band": "Activity band",
        },
        title="Group-level observed activity (not individual engagement)",
        color_discrete_map={
            "lower_observed_activity": "#d97706",
            "moderate_observed_activity": "#2563eb",
            "higher_observed_activity": "#15803d",
        },
    )
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(legend_title_text="Observed activity band")
    st.plotly_chart(fig, width="stretch", key=chart_key)
    if "explanation" in timeline.columns:
        with st.expander("Scoring explanations"):
            st.dataframe(
                timeline[["start_seconds", "activity_score", "explanation"]],
                width="stretch",
                hide_index=True,
            )


def _render_attendance(data: pd.DataFrame | None) -> None:
    st.subheader("Estimated visible presence")
    if not _is_usable(
        data,
        {
            "observed_presence_seconds",
            "required_presence_seconds",
        },
    ):
        st.info("Upload attendance records to see estimated presence.")
        return

    duration = pd.to_numeric(data["observed_presence_seconds"], errors="coerce")
    required = pd.to_numeric(data["required_presence_seconds"], errors="coerce")
    left, middle, right = st.columns(3)
    left.metric("Temporary tracks", len(data))
    middle.metric(
        "Mean estimated presence",
        f"{duration.mean():.1f} s" if duration.notna().any() else "Unavailable",
    )
    if "meets_presence_duration_threshold" in data.columns:
        meets = data["meets_presence_duration_threshold"].astype(str).str.lower()
        right.metric("Tracks meeting duration threshold", int(meets.isin(["true", "1"]).sum()))
    else:
        right.metric(
            "Configured presence threshold",
            f"{required.dropna().iloc[0]:.1f} s"
            if required.notna().any()
            else "Unavailable",
        )

    display_columns = [
        column
        for column in (
            "temporary_tracking_id",
            "tracking_id",
            "observed_presence_seconds",
            "required_presence_seconds",
            "meets_presence_duration_threshold",
            "observation_count",
            "presence_bouts",
        )
        if column in data.columns
    ]
    st.dataframe(data[display_columns], width="stretch", hide_index=True)


def _render_suggestions(data: pd.DataFrame | None) -> None:
    st.subheader("Instructor suggestions")
    if data is None:
        st.info("Upload a teaching-suggestions CSV to see recommendations.")
        return
    if data.empty:
        st.info("No suggestions were generated for this session.")
        return
    if not _is_usable(data, {"suggestion_type", "suggestion"}):
        return
    for _, item in data.iterrows():
        start = item.get("start_seconds", None)
        end = item.get("end_seconds", None)
        title = str(item["suggestion_type"]).replace("_", " ").title()
        if pd.notna(start) and pd.notna(end):
            title += f" · {float(start):.0f}–{float(end):.0f}s"
        with st.expander(title):
            if "observation" in data.columns and pd.notna(item.get("observation")):
                st.write(item["observation"])
            st.write(item["suggestion"])
            if "evidence" in data.columns and pd.notna(item.get("evidence")):
                st.caption(f"Evidence: {item['evidence']}")
            if "confidence" in data.columns and pd.notna(item.get("confidence")):
                st.caption(f"Confidence: {item['confidence']}")


def _render_student_analytics(
    participants: pd.DataFrame | None,
    timeline: pd.DataFrame | None,
    summary: pd.DataFrame | None,
) -> None:
    st.subheader("Session reflection")
    if summary is not None and not summary.empty:
        first_summary = summary.iloc[0]
        if "summary" in summary.columns:
            st.info(str(first_summary["summary"]))
        count_cols = [
            column
            for column in (
                "lower_activity_window_count",
                "moderate_activity_window_count",
                "higher_activity_window_count",
                "unscored_window_count",
            )
            if column in summary.columns
        ]
        if count_cols:
            st.dataframe(summary[count_cols], width="stretch", hide_index=True)

    if timeline is not None:
            _render_activity_timeline(timeline, chart_key="student_group_timeline")
    elif summary is None:
        st.info("Upload a group activity timeline or summary for session context.")

    if participants is None:
        st.info(
            "Upload student session analytics to review estimated presence. "
            "Individual activity consistency is unavailable with group-level scores."
        )
        return
    if participants.empty:
        st.info("No participant summaries are available.")
        return
    if not _is_usable(participants, {"observed_presence_seconds"}):
        return
    st.markdown("#### Estimated presence by temporary track")
    display_columns = [
        column
        for column in (
            "temporary_tracking_id",
            "observed_presence_seconds",
            "meets_presence_duration_threshold",
            "individual_activity_consistency",
            "reflection_note",
        )
        if column in participants.columns
    ]
    st.dataframe(
        participants[display_columns],
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Session Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Multimodal Session Dashboard")
    st.caption("A local, optional visualization layer for session CSV outputs.")
    st.warning(
        "These signals are indirect. Temporary tracks are not verified identities; "
        "activity scores do not establish engagement, attention, or learning. "
        "Review all outputs with human judgment."
    )

    with st.sidebar:
        st.header("Session data")
        demo_default = st.query_params.get("demo", "") == "1"
        demo_mode = st.checkbox(
            "Load synthetic demonstration data",
            value=demo_default,
            help="Synthetic values illustrate the dashboard and are not real session data.",
        )
        st.caption(
            f"Without an upload, the dashboard checks `{OUTPUT_DIRECTORY}` "
            "for standard output CSVs."
        )
        uploaded = {
            key: st.file_uploader(label, type=["csv"], key=f"upload_{key}")
            for key, (label, _) in UPLOADS.items()
        }

    if demo_mode:
        data = _demo_data()
        st.info(
            "Synthetic demonstration data is active. It does not represent "
            "students or a real classroom session."
        )
    else:
        data = {
            key: _load_upload_or_output(uploaded[key], filename)
            for key, (_, filename) in UPLOADS.items()
        }

    scores = data["scores"]
    if scores is None:
        scores = data["group_timeline"]

    tabs = st.tabs(
        [
            "Overview",
            "Attendance",
            "Activity timeline",
            "Suggestions",
            "Student analytics",
        ]
    )
    with tabs[0]:
        st.subheader("Session overview")
        score_count = (
            int(pd.to_numeric(scores["activity_score"], errors="coerce").notna().sum())
            if scores is not None and "activity_score" in scores.columns
            else 0
        )
        suggestion_count = len(data["suggestions"]) if data["suggestions"] is not None else 0
        attendance_count = len(data["attendance"]) if data["attendance"] is not None else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Attendance track records", attendance_count)
        col2.metric("Scored activity windows", score_count)
        col3.metric("Instructor suggestions", suggestion_count)
        if score_count:
            _render_activity_timeline(scores, chart_key="overview_activity_timeline")
        else:
            st.info(
                "No activity timeline is available yet. Upload a CSV in the "
                "sidebar or run the processing pipeline to create outputs."
            )
    with tabs[1]:
        _render_attendance(data["attendance"])
    with tabs[2]:
        _render_activity_timeline(scores, chart_key="activity_tab_timeline")
    with tabs[3]:
        _render_suggestions(data["suggestions"])
    with tabs[4]:
        _render_student_analytics(
            data["participants"],
            data["group_timeline"],
            data["group_summary"],
        )


if __name__ == "__main__":
    main()
