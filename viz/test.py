import PySimpleGUI as gui
import train_util
from utils.config import get_dataset_config_by_name
from structs import Anno, Sample
from typing import List


def get_sorted_annos(annos: List[Anno]) -> List[Anno]:
    return sorted(annos, key=lambda anno: anno.begin_offset)


def print_colored_text(sample: Sample, multiline: gui.Multiline):
    sample_text = sample.text
    sorted_annos = get_sorted_annos(sample.annos.gold)
    curr_offset = 0
    for anno in sorted_annos:
        text_before_anno = sample_text[curr_offset:anno.begin_offset]
        anno_text = sample_text[anno.begin_offset:anno.end_offset]
        assert anno_text == anno.extraction, f"sample_text: {anno_text}, extraction: {anno.extraction}"
        multiline.print(text_before_anno, end='')
        multiline.print(anno_text, colors='red', end='')
        curr_offset += len(text_before_anno) + len(anno_text)
    if curr_offset < len(sample_text):
        multiline.print(sample_text[curr_offset:], end='')


def main():
    train_samples = train_util.get_train_samples(get_dataset_config_by_name('cdr_vanilla'))
    curr_idx = 0
    layout = [
        [gui.Multiline(train_samples[curr_idx].text, key='text', size=(500, None), font=('arial', 10))],
        [gui.Button("OK")]
    ]
    window = gui.Window(title='Annotation Visualizer', layout=layout, size=(600, 600), text_justification='left')
    while True:
        event, values = window.read()
        print(event)
        if event == 'OK':
            print("Next sample")
            curr_idx += 1
            window['text'].update('')
            print_colored_text(sample=train_samples[curr_idx], multiline=window['text'])
        if event == gui.WIN_CLOSED or event == 'Cancel':
            break
    window.close()


main()
