# coding=utf-8

from __future__ import print_function
import sys, json
# import imp
# imp.reload(sys)
# sys.setdefaultencoding('utf-8')



def main():
    json_file = sys.argv[1]

    dataset = json.load(open(json_file))
    final = []
    for d in dataset[: 400]:
      final.append(d)
    json.dump(final, open('conala-dev.json', 'w'), indent = 2)

    # with open(seq_input, 'w') as f_inp, open(seq_output, 'w') as f_out:
    #     for example in dataset:
    #         f_inp.write(' '.join(example['intent_tokens']) + '\n')
    #         f_out.write(' '.join(example['snippet_tokens']) + '\n')


if __name__ == '__main__':
    main()
