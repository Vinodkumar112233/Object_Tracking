# TODO

## Tracking label correction
- [x] Add stable per-track class assignment (avoid per-frame flips)
- [x] Add class filtering/remap for vehicle-like categories
- [x] Tune thresholds so low-confidence mismatches don’t override the track label
- [x] Run smoke tests (`pytest tests/`) 
- [ ] Run quick manual test (`python src/tracker.py --source 0 --model yolov8n.pt --conf 0.4`)


