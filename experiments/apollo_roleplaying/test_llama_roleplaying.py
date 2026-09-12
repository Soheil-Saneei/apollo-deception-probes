import unittest

from run_llama import encode,DATE


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['date_string']==DATE
        return '[assistant]'

    def __call__(self,text,**kwargs):
        return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])

    def decode(self,ids,**kwargs):
        return ''.join(map(chr,ids))


class LlamaRoleplayingTests(unittest.TestCase):
    def test_exact_completion_and_split_preserved(self):
        row=dict(id='sample',messages=[],completion='  "An exact answer."\n',split='test',label='honest')
        encoded=encode(CharacterTokenizer(),row)
        self.assertEqual(encoded['split'],row['split'])
        self.assertEqual(encoded['completion'],row['completion'])
        positions=encoded['completion_positions']
        self.assertEqual(CharacterTokenizer().decode([encoded['input_ids'][i] for i in positions]),row['completion'])
        self.assertEqual(len(positions),len(row['completion']))

    def test_straddling_token_rejected(self):
        class Straddling(CharacterTokenizer):
            def __call__(self,text,**kwargs):
                return dict(input_ids=[1,2],offset_mapping=[(0,10),(10,len(text))])
        with self.assertRaisesRegex(ValueError,'straddles'):
            encode(Straddling(),dict(id='sample',messages=[],completion='answer'))


if __name__=='__main__':
    unittest.main()
