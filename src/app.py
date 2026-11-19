from pathlib import Path
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer,AutoModel,AutoImageProcessor,ViTModel,CLIPModel,CLIPProcessor
try:
 from .models import MODELS
 from .data import LABELS
except ImportError:
 from models import MODELS
 from data import LABELS

device='cuda' if torch.cuda.is_available() else 'cpu'
heads={}
for name,cls in MODELS.items():
 p=Path(f'checkpoints/{name}.pt')
 if not p.exists():raise FileNotFoundError(f'Missing {p}; train first.')
 model=cls().to(device);model.load_state_dict(torch.load(p,map_location=device,weights_only=True)['state_dict']);model.eval();heads[name]=model
rt=AutoTokenizer.from_pretrained('FacebookAI/roberta-base');roberta=AutoModel.from_pretrained('FacebookAI/roberta-base').to(device).eval()
vi=AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k');vit=ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device).eval()
cp=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32');clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()

def predict(image,text):
 if image is None or not text.strip():raise gr.Error('Upload an image and type text.')
 image=image.convert('RGB')
 with torch.inference_mode():
  r=rt([text],return_tensors='pt',padding=True,truncation=True,max_length=64).to(device)
  v=vi(images=[image],return_tensors='pt').to(device)
  c=cp(text=[text],images=[image],return_tensors='pt',padding='max_length',truncation=True,max_length=64).to(device)
  t=clip.text_projection(clip.text_model(input_ids=c['input_ids'],attention_mask=c['attention_mask']).last_hidden_state)
  im=clip.visual_projection(clip.vision_model(pixel_values=c['pixel_values']).last_hidden_state)
  features={'roberta':roberta(**r).last_hidden_state[:,0],'vit':vit(**v).last_hidden_state[:,0],'clip_text':t,'clip_image':im,'clip_mask':c['attention_mask']}
  result={}
  for name,head in heads.items():
   probs=head(features).softmax(-1)[0].cpu().tolist()
   result[name]={LABELS[i]:float(probs[i]) for i in range(3)}
 return result['text'],result['image'],result['cross'],result['concat']

with gr.Blocks(title='ImplicitMABSA') as demo:
 gr.Markdown('# ImplicitMABSA — Masked-Aspect Sentiment Demo\nUpload an image and enter a sentence with the target aspect removed or replaced by `[MASKED_ASPECT]`.')
 with gr.Row():
  image=gr.Image(type='pil',label='Tweet image');text=gr.Textbox(lines=4,label='Masked sentence',placeholder='The [MASKED_ASPECT] was absolutely amazing!')
 button=gr.Button('Predict sentiment',variant='primary')
 with gr.Row():
  t=gr.Label(label='RoBERTa text-only');i=gr.Label(label='ViT image-only')
 with gr.Row():
  m=gr.Label(label='CLIP cross-attention');c=gr.Label(label='CLIP concatenation')
 button.click(predict,[image,text],[t,i,m,c])
if __name__=='__main__':demo.launch(share=False)
