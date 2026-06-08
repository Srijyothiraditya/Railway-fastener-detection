# Railway Track Fastener Detection

## Overview
Automated railway track inspection using YOLOv8 instance segmentation.
Detects fasteners and sleepers and classifies track condition into 3 categories.

## Track Condition Categories
| Category | Label | Description |
|----------|-------|-------------|
| 1 | GOOD | Sleeper present, fasteners on BOTH sides |
| 2 | WARNING | Sleeper present, fastener on ONE side only |
| 3 | CRITICAL | Sleeper present, NO fasteners on either side |

## Model Performance
| Metric | Score |
|--------|-------|
| mAP@50 | 79.37% |
| mAP@50-95 | 65.12% |
| Precision | 95.72% |
| Recall | 72.42% |

## Test Set Results (41 images)
| Category | Count | Percentage |
|----------|-------|------------|
| GOOD | 22 | 53% |
| WARNING | 19 | 46% |
| CRITICAL | 0 | 0% |

## Dataset
- 1066 total images (984 train / 41 val / 41 test)
- Classes: Fastener, Sleeper
- Annotation: Roboflow polygon segmentation

## Usage
pip install ultralytics

Train: python scripts/train_v2.py --device 0

Assess: python scripts/assess_condition.py --source path/to/images --device 0
