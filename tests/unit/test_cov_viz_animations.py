"""Behavioural tests for active_inference.visualization.animations."""

import pytest

from active_inference.visualization.animations import (
    AnimationEngine,
    AnimationFrame,
    AnimationSequence,
    AnimationType,
    ProcessAnimation,
)


def _frames(*timestamps):
    return [
        AnimationFrame(frame_id=i, timestamp=t, components={"i": i})
        for i, t in enumerate(timestamps)
    ]


def test_animation_frame_defaults():
    frame = AnimationFrame(frame_id=0, timestamp=0.0, components={})
    assert frame.narration is None
    assert frame.highlights == []
    assert AnimationFrame(0, 0.0, {}).highlights is not frame.highlights


class TestAnimationSequence:
    def test_frame_lookup_and_clamping(self):
        seq = AnimationSequence("s", "S", "d", _frames(0, 1, 2, 3, 4), duration=5.0)
        assert seq.frame_rate == 30
        assert seq.get_frame_at_time(0).frame_id == 0
        assert seq.get_frame_at_time(2.5).frame_id == 2
        assert seq.get_frame_at_time(4.99).frame_id == 4
        assert seq.get_frame_at_time(100).frame_id == 4
        assert seq.get_frame_at_time(-3).frame_id == 0

    def test_empty_sequence_has_no_frame(self):
        assert AnimationSequence("s", "S", "d", [], 1.0).get_frame_at_time(0) is None

    def test_zero_duration_sequence_returns_its_frame(self):
        seq = AnimationSequence("s", "S", "d", _frames(0.0), duration=0.0)
        assert seq.get_frame_at_time(0.0).frame_id == 0
        assert seq.get_frame_at_time(3.0).frame_id == 0


class TestProcessAnimation:
    def test_perception_action_cycle(self):
        seq = ProcessAnimation().create_perception_action_cycle()
        assert seq.id == "perception_action_cycle"
        assert seq.duration == 5.0
        assert [f.frame_id for f in seq.frames] == [0, 1, 2, 3, 4]
        assert seq.frames[0].components["sensory_input"]["active"] is False
        assert seq.frames[1].highlights == ["sensory_input"]
        assert seq.frames[2].components["prediction"]["value"] == 0.7
        assert seq.frames[3].components["prediction_error"]["value"] == 0.1
        assert seq.frames[4].components["action"]["selection"] == "action_1"
        assert seq.frames[4].narration.startswith("Action is selected")

    def test_belief_updating_defaults(self):
        seq = ProcessAnimation().create_belief_updating_animation()
        assert seq.id == "belief_updating"
        assert len(seq.frames) == 8
        assert seq.frames[0].components["observation"] == [0, 0, 0]
        assert seq.frames[1].components["observation"] == [0.8, 0.1, 0.1]
        assert seq.frames[2].components["prediction_error"] == pytest.approx(
            [0.3, -0.2, -0.1]
        )

        first_update = seq.frames[3]
        assert first_update.components["learning_rate"] == 0.1
        assert first_update.components["updated_belief"] == pytest.approx(
            [0.53, 0.28, 0.19]
        )
        assert first_update.narration == "Belief update with learning rate α=0.1"
        assert [f.timestamp for f in seq.frames[3:]] == [3.0, 3.5, 4.0, 4.5, 5.0]
        for frame in seq.frames[3:]:
            assert sum(frame.components["updated_belief"]) == pytest.approx(1.0)

    def test_belief_updating_custom_inputs_are_normalised(self):
        seq = ProcessAnimation().create_belief_updating_animation(
            initial_belief=[0.5, 0.5], observations=[1.0, 0.0]
        )
        last = seq.frames[-1].components
        assert last["learning_rate"] == 0.9
        assert last["updated_belief"] == pytest.approx([0.95, 0.05])
        assert seq.frames[0].components["belief_distribution"] == [0.5, 0.5]

    def test_multi_agent_frames_are_snapshots(self):
        seq = ProcessAnimation().create_multi_agent_animation()
        assert seq.id == "multi_agent_interaction"
        assert len(seq.frames) == 5

        initial = seq.frames[0].components["agents"]
        assert len(initial) == 3
        assert initial[0] == {"beliefs": [0.6, 0.3, 0.1], "action": None, "reward": 0.0}
        assert seq.frames[1].components["environment"]["state"] == "changed"

        updated = seq.frames[2].components["agents"][0]
        assert updated["beliefs"] == pytest.approx([0.8, 0.2, 0.0])
        assert updated["action"] is None

        messages = seq.frames[3].components["communication"]["messages"]
        assert [(m["sender"], m["receiver"]) for m in messages] == [(0, 1), (1, 2)]
        assert messages[0]["content"] == pytest.approx([0.8, 0.2, 0.0])

        final = seq.frames[4].components["agents"]
        assert all(agent["action"] == "coordinate" for agent in final)

    @pytest.mark.parametrize("num_agents, expected_pairs", [(1, []), (2, [(0, 1)])])
    def test_multi_agent_small_groups(self, num_agents, expected_pairs):
        seq = ProcessAnimation().create_multi_agent_animation(num_agents=num_agents)
        assert len(seq.frames[0].components["agents"]) == num_agents
        messages = seq.frames[3].components["communication"]["messages"]
        assert [(m["sender"], m["receiver"]) for m in messages] == expected_pairs

    def test_neural_network_frames_track_progress(self):
        seq = ProcessAnimation().create_neural_network_animation()
        assert seq.id == "neural_network_ai"
        assert len(seq.frames) == 6

        start = seq.frames[0].components
        assert start["input_layer"] == [0.0] * 4
        assert start["hidden_layers"] == [[0.0] * 8, [0.0] * 8]
        assert start["output_layer"] == [0.0] * 4
        assert (start["weights"], start["activation"]) == ("initial", "none")

        assert seq.frames[1].components["activation"] == "forward"
        assert seq.frames[1].components["hidden_layers"][0] == [0.0] * 8
        assert seq.frames[2].components["hidden_layers"] == [[0.5] * 8, [0.5] * 8]
        assert seq.frames[2].components["output_layer"] == [0.0] * 4
        assert seq.frames[3].components["output_layer"] == [0.7, 0.2, 0.05, 0.05]
        assert seq.frames[3].components["weights"] == "initial"
        assert seq.frames[4].components["weights"] == "updating"
        assert seq.frames[5].components["activation"] == "minimizing"

    def test_neural_network_custom_layers(self):
        seq = ProcessAnimation().create_neural_network_animation(layers=[2, 3, 5])
        start = seq.frames[0].components
        assert start["input_layer"] == [0.0, 0.0]
        assert start["hidden_layers"] == [[0.0] * 3]
        assert start["output_layer"] == [0.0] * 5
        assert seq.frames[2].components["hidden_layers"] == [[0.5] * 3]

    def test_information_flow_frames_track_progress(self):
        seq = ProcessAnimation().create_information_flow_animation()
        assert seq.id == "information_flow"
        assert seq.duration == 8.0
        assert len(seq.frames) == 8

        start = seq.frames[0].components
        assert start["sensory_input"] == {"value": 0.0, "entropy": 1.0}
        assert start["free_energy"] == {"value": 1.0, "trend": "decreasing"}
        assert start["action"]["selection"] is None

        assert seq.frames[1].components["sensory_input"] == {
            "value": 0.8,
            "entropy": 0.3,
        }
        assert seq.frames[1].components["generative_model"]["accuracy"] == 0.3
        assert seq.frames[2].components["generative_model"]["accuracy"] == 0.6
        assert seq.frames[3].components["prediction"]["information"] == 0.4
        assert seq.frames[4].components["prediction_error"]["magnitude"] == 0.2
        assert seq.frames[5].components["free_energy"]["value"] == 0.7
        assert seq.frames[6].components["action"]["selection"] == "optimal"
        assert seq.frames[7].components["free_energy"] == {
            "value": 0.4,
            "trend": "stable",
        }

    def test_free_energy_minimization(self):
        seq = ProcessAnimation().create_free_energy_minimization()
        assert seq.id == "free_energy_minimization"
        assert len(seq.frames) == 11

        values = [f.components["free_energy"]["value"] for f in seq.frames]
        assert values[0] == 1.0
        assert values[-1] == pytest.approx(0.2)
        assert all(a > b for a, b in zip(values, values[1:]))

        last = seq.frames[-1]
        assert last.timestamp == 5.0
        assert last.components["free_energy"]["progress"] == pytest.approx(0.8 / 0.9)
        assert last.components["model_updates"] == {"count": 10, "active": False}
        assert seq.frames[9].components["model_updates"]["active"] is True
        assert last.components["prediction_accuracy"]["value"] == pytest.approx(0.9)
        assert last.components["belief_confidence"]["value"] == pytest.approx(0.9)
        assert last.narration == "Step 10: Free energy = 0.200"
        assert seq.frames[0].highlights == ["free_energy"]
        assert seq.frames[1].highlights == []


class TestAnimationEngine:
    def test_create_animation_registers_sequence(self):
        engine = AnimationEngine({"fps": 30})
        assert engine.config == {"fps": 30}
        frames = _frames(0.0, 1.0, 2.5)
        animation = engine.create_animation(
            AnimationType.STEP_BY_STEP, "My Anim", frames
        )
        assert animation.id == "step_by_step_my_anim"
        assert animation.description == "step_by_step animation: My Anim"
        assert animation.duration == 2.5
        assert engine.animations == {"step_by_step_my_anim": animation}

    def test_create_empty_animation(self):
        animation = AnimationEngine({}).create_animation(
            AnimationType.CONTINUOUS, "Empty", []
        )
        assert animation.duration == 0.0
        assert animation.frames == []

    def test_get_process_animation(self, caplog):
        engine = AnimationEngine({})
        assert (
            engine.get_process_animation("perception_action_cycle").id
            == "perception_action_cycle"
        )
        assert (
            engine.get_process_animation("free_energy_minimization").id
            == "free_energy_minimization"
        )
        assert engine.get_process_animation("unknown") is None
        assert "Process animation not found: unknown" in caplog.text

    def test_play_animation(self):
        engine = AnimationEngine({})
        engine.create_animation(AnimationType.DEMONSTRATION, "Demo", _frames(0, 2, 4))
        assert engine.play_animation("missing") == {}
        assert engine.play_animation("demonstration_demo", speed=2.0) == {
            "animation_id": "demonstration_demo",
            "name": "Demo",
            "total_frames": 3,
            "duration": 2.0,
            "frame_rate": 60.0,
            "is_playing": True,
        }

    def test_get_frame_at_time(self):
        engine = AnimationEngine({})
        engine.create_animation(AnimationType.TRANSITION, "T", _frames(0, 2, 4))
        engine.create_animation(AnimationType.TRANSITION, "Empty", [])
        engine.create_animation(AnimationType.TRANSITION, "Single", _frames(0.0))

        assert engine.get_frame_at_time("missing", 0) is None
        assert engine.get_frame_at_time("transition_empty", 0) is None
        assert engine.get_frame_at_time("transition_single", 0)["frame_id"] == 0
        assert engine.get_frame_at_time("transition_t", 2.0) == {
            "frame_id": 1,
            "timestamp": 2,
            "components": {"i": 1},
            "narration": None,
            "highlights": [],
        }

    def test_export_animation(self):
        engine = AnimationEngine({})
        frames = _frames(0, 1)
        frames[1].narration = "end"
        frames[1].highlights = ["i"]
        engine.create_animation(AnimationType.INTERACTIVE, "X", frames)

        assert engine.export_animation("missing") is None
        assert engine.export_animation("interactive_x", format="gif") is None
        exported = engine.export_animation("interactive_x")
        assert exported["id"] == "interactive_x"
        assert exported["duration"] == 1
        assert exported["frame_rate"] == 30
        assert exported["frames"][1] == {
            "frame_id": 1,
            "timestamp": 1,
            "components": {"i": 1},
            "narration": "end",
            "highlights": ["i"],
        }
