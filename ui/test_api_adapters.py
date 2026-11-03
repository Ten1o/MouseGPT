"""Exercise provider operations with fake HTTP replies, never live credentials."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import worker


class ProviderAdapters(unittest.TestCase):
    def test_describe_and_embed(self):
        from PIL import Image
        fake={k:'Observed fact.' for k in ['Overall','Head','Limb','Torso','Others','Keywords']}
        class Response:
            status_code=200
            text=''
            def json(self):return {'choices':[{'message':{'content':json.dumps(fake)}}]}
        env={f'MOUSEGPT_{r}_{k}':v for r in ['VISION','EMBEDDING'] for k,v in {'PROVIDER':'openai','ENDPOINT':'https://api.openai.com/v1','API_KEY':'test-key-not-real','MODEL':'text-embedding-3-large' if r=='EMBEDDING' else 'gpt-4o'}.items()}
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,env):
            root=Path(tmp)
            for cam in (3,5):
                folder=root/f'data{cam}';folder.mkdir();Image.new('RGB',(32,32)).save(folder/'00000000.jpg')
            (root/'velocity.json').write_text(json.dumps({'0':dict.fromkeys(['base','head','forepawL','forepawR','hindpawL','hindpawR'],1)}))
            f={'crop3':str(root/'data3'),'crop5':str(root/'data5'),'velocity':str(root/'velocity.json'),'out2':str(root/'output'),'prompt':'Describe facts.','limit':1}
            with patch('requests.post',return_value=Response()) as post:
                result=worker.run('describe',f,{})
                self.assertEqual(post.call_count,1)
                payload=post.call_args.kwargs['json']
                self.assertEqual(payload['messages'][0]['content'],'Describe facts.')
                self.assertEqual(len(payload['messages'][1]['content']),3)
            self.assertEqual(json.loads((root/'output/descriptions/00000000.json').read_text()),fake)
            class EmbeddingResponse:
                status_code=200
                def raise_for_status(self):pass
                def json(self):return {'data':[{'embedding':[0.1]*3072}]}
            with patch('requests.post',return_value=EmbeddingResponse()) as post:
                result=worker.run('embed',{'descriptions':str(root/'output/descriptions'),'features':str(root/'vectors'),'limit':1},{})
                self.assertEqual(post.call_count,1)
            self.assertTrue((root/'vectors/00000000_des.csv').is_file())
            worker.run('previews',f,{})
            with Image.open(root/'output/paired_previews/00000000.jpg') as image:self.assertEqual(image.size,(64,32))


if __name__=='__main__':unittest.main()
