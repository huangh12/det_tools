# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import chips
from ..bbox.bbox_transform import clip_boxes, ignore_overlaps
import numpy as np


class chip_generator(object):
    def __init__(self, chip_stride=32, use_cpp=True):
        self.use_cpp = use_cpp
        self.chip_stride = chip_stride

    def generate(self, boxes, width, height, chipsize):
        if self.use_cpp:
            return self._cgenerate(boxes, width, height, chipsize, self.chip_stride)
        else:
            return self._pygenerate(boxes, width, height, chipsize, self.chip_stride)

    @staticmethod
    def _cgenerate(boxes, width, height, chipsize, stride):
        boxes = clip_boxes(boxes, np.array([height - 1, width - 1]))
        return chips.generate(np.ascontiguousarray(boxes, dtype=np.float32),
                              width, height, chipsize, stride)

    @staticmethod
    def _pygenerate(boxes, width, height, chipsize, stride):
        chips = []
        boxes = clip_boxes(boxes, np.array([height-1, width-1]))
        # ensure coverage of image for worst case
        # corners
        if height < chipsize and width < chipsize:
            chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
        elif height > chipsize and width < chipsize:
            chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
            chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
        elif height < chipsize and width > chipsize:
            chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
            chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
        else:
            chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
            chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
            chips.append([max(width - chipsize, 0), max(height - chipsize, 0), width-1, height-1])            

        for i in range(0, width - int(chipsize), stride):
            for j in range(0, height - int(chipsize), stride):
                x1 = i
                y1 = j
                x2 = i + chipsize - 1
                y2 = j + chipsize - 1
                chips.append([x1, y1, x2, y2])

        for j in range(0, height - int(chipsize), stride):
            x1 = max(width - chipsize - 1,0)
            y1 = j
            x2 = width - 1
            y2 = j + chipsize - 1
            chips.append([x1, y1, x2, y2])

        for i in range(0, width - int(chipsize), stride):
            x1 = i
            y1 = max(height - chipsize - 1,0)
            x2 = i + chipsize - 1
            y2 = height - 1
            chips.append([x1, y1, x2, y2])

        chips = np.array(chips).astype(np.float)

        p = np.random.permutation(chips.shape[0])
        chips = chips[p]

        overlaps = ignore_overlaps(chips, boxes.astype(np.float))
        chip_matches = []
        num_matches = []
        for j in range(len(chips)):
            nvids = np.where(overlaps[j, :] == 1)[0]
            chip_matches.append(set(nvids.tolist()))
            num_matches.append(len(nvids))

        # ------------------------------------------------------#
        # specially designed to pick the lost boxes (may big or small) #
        # however, after pick the lost boxes, the size of chip turns unexpected,
        # so batch size may decrease to fit into gpu for res50 or some strategies
        # is needed to fix that!

        # the lost boxes may uncover that the current ISN crop metric (scale=sqrt(w*h)) is suboptimal
        # One may want to crop according to the ratio or max(height, width)
        #-------------------------------------------------------#
        PICK_LOST = False
        DEBUG = False
        fchips = []
        
        if PICK_LOST:
            fullset = set(range(boxes.shape[0]))
            covered = set()
            for s in chip_matches:
                covered = covered | s

            tobecovered = fullset - covered
            if len(tobecovered) > 0:
                temp = boxes[list(tobecovered)]
                if DEBUG:
                    print('{} boxes uncovered!'.format(len(tobecovered)))
                    print('The area scales are: {}'.format(np.sqrt((temp[:,3]-temp[:,1])*(temp[:,2]-temp[:,0]))))
                    print('The uncovered boxes:\n',temp)

                    print('Add the chips to cover them...')
                addchips = temp + np.array([-16,-16,16,16])
                # addchips[:,0::2] = np.clip(addchips[:,0::2], 0, width-1)
                # addchips[:,1::2] = np.clip(addchips[:,1::2], 0, height-1)
                addchips = clip_boxes(addchips, np.array([height-1, width-1]))

                if DEBUG:
                    print(addchips)
                    print('the added chip (width,height):', list(zip((addchips[:,2]-addchips[:,0]),(addchips[:,3]-addchips[:,1]))))
                    print('the added chip scale:', np.sqrt((addchips[:,3]-addchips[:,1])*(addchips[:,2]-addchips[:,0])))
                    print('---------------------------------')

                for c in addchips:
                    fchips.append(c)
        #-------------------------------------------------------#

        # fchips = []
        totalmatches = 0
        while True:
            max_matches = 0
            max_match = max(num_matches)
            mid = np.argmax(np.array(num_matches))
            if max_match == 0:
                break
            if max_match > max_matches:
                max_matches = max_match
                maxid = mid
            bestchip = chip_matches[maxid]
            fchips.append(chips[maxid])
            totalmatches = totalmatches + max_matches

            # now remove all rois in bestchip
            for j in range(len(num_matches)):
                chip_matches[j] = chip_matches[j] - bestchip
                num_matches[j] = len(chip_matches[j])

        return fchips


# for dbg
if __name__ == "__main__":
    boxes, width, height, chipsize, stride = np.random.random([10,4]), 16, 16, 10, 2

    chips = []
    # ensure coverage of image for worst case
    # corners
    if height < chipsize and width < chipsize:
        chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
    elif height > chipsize and width < chipsize:
        chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
        chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
    elif height < chipsize and width > chipsize:
        chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
        chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
    else:
        chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
        chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
        chips.append([max(width - chipsize, 0), max(height - chipsize, 0), width-1, height-1])    

    print(chips)
    import pdb; pdb.set_trace()

    for i in range(0, width - int(chipsize), stride):
        for j in range(0, height - int(chipsize), stride):
            x1 = i
            y1 = j
            x2 = i + chipsize - 1
            y2 = j + chipsize - 1
            chips.append([x1, y1, x2, y2])

    for j in range(0, height - int(chipsize), stride):
        x1 = max(width - chipsize - 1,0)
        y1 = j
        x2 = width - 1
        y2 = j + chipsize - 1
        chips.append([x1, y1, x2, y2])

    for i in range(0, width - int(chipsize), stride):
        x1 = i
        y1 = max(height - chipsize - 1,0)
        x2 = i + chipsize - 1
        y2 = height - 1
        chips.append([x1, y1, x2, y2])
