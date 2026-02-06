from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from transformers import BitsAndBytesConfig
import re


SYSTEM_PROMPT = """
Eres un asistente general experto en programación y tecnología.
Explicas conceptos de forma clara, sencilla y didáctica.
Respondes en español neutro.
Das ejemplos cuando es útil.
No inventes información.
"""

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Pad token configurado como EOS token.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
print("Modelo listo.")

class Prompt(BaseModel):
    message: str


@app.get("/")
def root():
    with open("static/chat.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/chat")
def chat(prompt: Prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt.message},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            ).to("cuda")


        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]


        output = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )


        generated_tokens = output[0][inputs.input_ids.shape[-1]:]


        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()


        response = re.sub(r'<\|reserved_special_token_\d+\|>', '', response)


        return JSONResponse(
            content={"response": response},
            media_type="application/json; charset=utf-8"
        )
    
    except torch.cuda.OutOfMemoryError as e:
        return JSONResponse(
            content={
                "error": "GPU sin memoria suficiente. Prueba con max_new_tokens más bajo (ej: 300/500), cierra otras aplicaciones que usen la GPU, o reinicia el servidor."
            },
            status_code=500
        )
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error inesperado: {error_msg}")
        return JSONResponse(
            content={
                "error": f"Error interno del servidor: {str(e)}. Revisa la consola del servidor para más detalles."
            },
            status_code=500
        )