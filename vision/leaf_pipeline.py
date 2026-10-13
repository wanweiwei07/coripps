"""Look, find the leaves, pick one, line the cutter up with it -- in one go.

1. survey (``capture_survey``): five fixed views of the plant, 'g' in the
   terminal starts them; the arm stays at the last view
2. leaves (``reconstruct_survey`` + ``leaf_frames``): the plane sweep, a
   frame per leaf, the leaf with the most room at its petiole
3. approach (``leaf_approach_sim``): the cutter tcp onto that leaf's frame,
   planned from the real current joints, looped in the viewer; 'g' in the
   viewer window runs it on the real sampler

The plant goes at ``capture_survey.VIEW_CENTER`` (sampler base frame).

    py -3.12 coripps/vision/leaf_pipeline.py
"""

import os
import sys
import json

from coripps.vision import capture_survey as cs
from coripps.vision import leaf_approach_sim as las


MIN_IMAGES = 4                  # the reference + 3 others (reconstruct MIN_VIEWS)


def main():
    print('== 1. survey')
    if not cs.main():
        return False
    if not os.path.exists(cs.RECORD_PATH):
        return False
    with open(cs.RECORD_PATH, encoding='utf-8') as f:
        n = len(json.load(f))
    if n < MIN_IMAGES:
        print(f'only {n} images -- too few for the leaves')
        return False
    print('\n== 2-3. leaves, choice, approach')
    return las.main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
