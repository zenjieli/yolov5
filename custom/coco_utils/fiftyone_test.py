import fiftyone as fo
import fiftyone.zoo as foz

def download():
    dataset = foz.load_zoo_dataset('coco-2017',
                    splits=['train', 'validation'],
                    label_types=["detections"],
                    max_samples=2,
                    shuffle=True,
                    classes=["person"])
    session = fo.launch_app(dataset)

if __name__ == '__main__':
    download()