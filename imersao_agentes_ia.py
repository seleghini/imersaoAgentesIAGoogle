"""
AULA 01 - Criando um agente de triagem com Gemini 2.5

Objetivo: Criar um agente de triagem para Service Desk utilizando o modelo Gemini 2.5 da Google.
O agente deve ser capaz de classificar as solicitações dos usuários em três categorias: AUTO_RESOLVER, PEDIR_INFO e ABRIR_CHAMADO.
Além disso, o agente deve determinar a urgência da solicitação (ALTA, MEDIA, BAIXA) e identificar quaisquer campos faltantes na solicitação.
Também será realizado um teste do prompt do sistema para garantir que o agente está funcionando conforme o esperado.
"""

#importando a biblioteca de logging
import logging

#config basico do logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] - %(message)s')

#importando a chave e o modelo de IA que será utilizado
from langchain_google_genai import ChatGoogleGenerativeAI  

#definindo com o LLM para testar o prompt do sistema
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

#conectando e criando o LLM
from langchain_core.messages import SystemMessage, HumanMessage

#importando a biblioteca para carregar as variáveis de ambiente
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pegando a chave da API e modelo do Gemini do arquivo .env
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL = os.getenv('MODEL')
temperature_str = os.getenv('TEMPERATURE')

#verificação se as chaves foram carregadas corretamente
if not GEMINI_API_KEY:
    raise ValueError("A variável de ambiente GEMINI_API_KEY não foi encontrada. Verifique seu arquivo .env")
if not MODEL:
    raise ValueError("A variável de ambiente MODEL não foi encontrada. Verifique seu arquivo .env")
if not temperature_str:
    raise ValueError("A variável de ambiente TEMPERATURE não foi encontrada. Verifique seu arquivo .env")

# Convertendo a temperatura para float
TEMPERATURE = float(temperature_str)

#preparando e conectando o modelo do Gemini que vamos utilizar
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=TEMPERATURE, api_key=GEMINI_API_KEY)

#testando o modelo do Gemini 2.5
#perg_test = "Qual a temperatura máxima que fará hoje em São Paulo?"
#resp_test = llm.invoke(perg_test)
#print("\nPergunta: "+perg_test)
#print("Resposta: "+resp_test.content+"\n")
#print("==============================================================================================================\n")

#criando o prompt do sistema
TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para politicas internas. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com: \n"
    "{\n"
    ' "decisão": "AUTO_RESOLVER" | "PEDIR INFO" | "ABRIR CHAMADO", \n'
    ' "urgencia": "ALTA" | "MEDIA" | "BAIXA", \n'
    ' "campos faltantes": ["..."]\n'
    "}\n"
    "Regras :\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex. "Posso reembolsar a internet do meu Home Office?")\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex. "PReciso de ajuda com uma politica","Tenho uma dúvida geral")\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado no sistema\n'
    "Analise a mensagem e decida a ação mais apropriada"
)

#exemplo de retorno do prompt
"""
    decisão: PEDIR_INFO
    urgencia: MEDIA
    campos faltantes: []
"""

class TriagemOut(BaseModel):
  decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
  urgencia: Literal["ALTA", "MEDIA", "BAIXA"]
  campos_faltantes: List[str] = Field(default_factory=list)

#criando uma nova LLM para a triagem
llm_triagem = ChatGoogleGenerativeAI(model=MODEL, temperature=TEMPERATURE, api_key=GEMINI_API_KEY)

#criando o fluxo de mensagem a triagem
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

#criando uma função para chamar a triagem
def triagem(mensagem: str) -> Dict:
  saida: TriagemOut = triagem_chain.invoke([
      SystemMessage(content=TRIAGEM_PROMPT),
      HumanMessage(content=mensagem)
  ])

  return saida.model_dump()

#criando as perguntas teste do prompt do agente
testes = ["Posso reembolsar a internet? ",
          "Quero mais 5 dia de trabalho remoto, como faço?",
          "Posso reembolsar cursos ou treinamentos da Alura?",
          "É possível reembolsar certificações do Google Cloud?",
          "Posso obter o Google Gemini de graça?",
          "Qual a palavra chave da aula de hoje?",
          "Quantas capivaras tem no Rio Pinheiros"]

#fazendo a iteração por For para chamar as perguntas do teste
#for msg_teste in testes:
#  print(f"Pergunta: {msg_teste}")
#  print(f"Resposta: {triagem(msg_teste)}\n")
#  print("==============================================================================================================\n")


"""
AULA 02 - Criando um agente de consulta a documentos com Gemini Embeddings
    Objetivo: Criar um agente de consulta a documentos utilizando o modelo Gemini Embeddings da Google.
    O agente deve ser capaz de carregar documentos em PDF, dividir esses documentos em pedaços menores, criar embeddings para esses pedaços e armazená-los em um vetor de busca.
    Em seguida, o agente deve ser capaz de recuperar informações relevantes dos documentos com base em consultas do usuário.  
"""

#importando a biblioteca para trabalhar com arquivos
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

#carregando os documentos da pasta Docs
PATH_DOC = os.getenv('PATH_DOC')

if not PATH_DOC:
    raise ValueError("A variável de ambiente PATH_DOC não foi encontrada. Verifique seu arquivo .env")

docs = []

for n in Path(PATH_DOC).glob("*.pdf"):
    try:
       loader = PyMuPDFLoader(str(n))
       docs.extend(loader.load())
       #print(f"Documento {n.name} carregado com sucesso.")
    except Exception as e:
       print(f"Erro ao carregar o Documento {n.name}: {e}")

if not docs:
    raise ValueError("Nenhum documento PDF foi encontrado ou carregado com sucesso. "
                     "Verifique o caminho em PATH_DOC e se os arquivos na pasta não estão corrompidos.")

#carregando biblioteca para quebrar os documentos em pedaços menores
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)


#for chunk in chunks:
#    print(chunk)
#    print("-------------------------------------------------------------------------------------------------------\n")

#carregando a biblioteca para criar os embeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#carregando a variavel de ambiente do modelo de embeddings
MODEL_EMBEDDINGS = os.getenv('MODEL_EMBEDDINGS')
if not MODEL_EMBEDDINGS:
    raise ValueError("A variável de ambiente MODEL_EMBEDDINGS não foi encontrada. Verifique seu arquivo .env")

embedings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS, google_api_key=GEMINI_API_KEY)

#carregando a biblioteca para criar e recuperar o vetor de busca
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embedings)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.3, "k":4})

#criando o agente de consulta aos documentos
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt_rag = ChatPromptTemplate.from_messages([(
   "system",
   "Você é um assistente de Políticas Internas (RH/TI) da empresa Carraro Desenvolvimento."
   "Responda SOMENTE com base no contexto fornecido."
   "Se não houver base suficiente, responda que não sabe."),

   ("human", "Pergunta : {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Formatadores
import re, pathlib

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]


#criando a função para perguntar ao agente de consulta aos documentos
def perguntar_politica_RAG(pergunta: str) -> Dict:
   docs_relacionados = retriever.invoke(pergunta)

   if not docs_relacionados:
      return {"answer": "Não sei", "citacoes": [], "contexto_encontrado": False}
   
   answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})

   txt = (answer or "").strip()

   if txt.strip(".?!") == "Não sei":
        return {"answer": "Não sei", "citacoes": [], "contexto_encontrado": False}
    
   return {"answer": txt, "citacoes": formatar_citacoes(docs_relacionados, pergunta), "contexto_encontrado": True}

#criando as perguntas para o agente 
testes02 = ["Posso reembolsar a internet ? ",
          "Quero mais 5 dia de trabalho remoto, como faço?",
          "Posso reembolsar cursos ou treinamentos da Alura?",
          "Quantas capivaras tem no Rio Pinheiros"]

#for msg_teste in testes02:
    #resposta = perguntar_politica_RAG(msg_teste)
    #print(f"PERGUNTA: {msg_teste}")
    #print(f"RESPOSTA: {resposta['answer']}\n")
    #if resposta['contexto_encontrado']:
        #print("CITAÇÕES")
        #for c in resposta['citacoes']:
            #print(f"- Documento: {c['documento']} - Página: {c['pagina']}")
            #print(f"  Trecho: {c['trecho']}\n")
        #print("=========================================================================================================")

"""
AULA 03 - Criando o agente de triagem + consulta a documentos com Gemini 2.5 + Embeddings
    Objetivo: Integrar o agente de triagem com o agente de consulta a documentos para criar um sistema completo de atendimento ao Service Desk.
    O agente deve ser capaz de classificar as solicitações dos usuários, determinar a urgência, identificar campos faltantes e, quando necessário, consultar documentos para fornecer respostas precisas.
"""

#importando as bibliotecas para criar o agente
from typing import TypedDict, Optional

#criando o tipo do estado do agente
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

#criando o nó de triagem
def node_triagem(state: AgentState) -> AgentState:
    logging.info("Executando nó de triagem...")
    return {**state, "triagem": triagem(state["pergunta"])}

#criando o nó de auto-resolver
def node_auto_resolver(state: AgentState) -> AgentState:
    logging.info("Executando nó de auto-resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes",[]),
        "rag_sucesso": resposta_rag["contexto_encontrado"]
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

#criando o nó de pedir mais informações
def node_pedir_info(state: AgentState) -> AgentState:
    logging.info("Executando nó de pedir info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    if(faltantes):
        detalhe = ','.join(faltantes) 
    else:
        detalhe = "Tema e contexto especifico"
    
    return {
            "resposta": f"Para avançar, preciso de mais detalhes: {detalhe}",
            "citacoes": [],
            "acao_final": "PEDIR_INFO"}

#criando o nó de abrir chamado
def node_abrir_chamado(state: AgentState) -> AgentState:
    logging.info("Executando nó de abrir chamado...")
    triagem = state["triagem"]

    return {
            "resposta": f"Chamado aberto com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
            "citacoes": [],
            "acao_final": "ABRIR_CHAMADO"}

#definição de palavras chaves para abertura de chamado
KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "acesso especial", "abrir ticket", "abrir chamado"]


#criando a função main do agente
def decidir_pos_triagem(state: AgentState) -> str:
    logging.info("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]
    
    if decisao == "AUTO_RESOLVER":
        return "auto"
    elif decisao == "PEDIR_INFO":
        return "info"
    elif decisao == "ABRIR_CHAMADO":
        return "chamado"

#criando a função para pós auto resolver
def decidir_pos_auto_resolver(state: AgentState) -> str:
    logging.info("Decidindo após o auto-resolver...")

    if state.get("rag_sucesso"):
        logging.info("RAG teve sucesso, finalizando o fluxo...")
        return "ok"
    
    state_da_pergunta = (state["pergunta"] or "").lower()
    
    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        logging.info("RAG Falhou, mas foram encontradas Palavra chave para abrir chamado, indo para abrir chamado...")
        return "chamado"

    logging.info("RAG Falhou, indo para pedir mais informações...")
    return "info"

#criando com o langgraph o fluxo do agente
from langgraph.graph import StateGraph, START, END 

workflow = StateGraph(AgentState)
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, { 
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)
grafo = workflow.compile()

#importando a biblioteca para desenhar e visualizar o grafo
from IPython.display import display, Image

graph_bytes = grafo.get_graph().draw_mermaid_png()
display(Image(graph_bytes))

#fazendo o FOR para pegar as perguntas do teste
for msg_test in testes:
    resposta_final = grafo.invoke({"pergunta": msg_test})

    triag = resposta_final.get("triagem", {})
    logging.info(f"PERGUNTA: {msg_test}")
    logging.info(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')} | CAMPOS FALTANTES: {triag.get('campos_faltantes',[])}")
    logging.info(f"RESPOSTA: {resposta_final.get('resposta')}\n")

    if resposta_final.get("citacoes"):
        logging.info("CITAÇÕES")
        for c in resposta_final.get("citacoes"):
            logging.info(f"Documento: {c['documento']} - Página: {c['pagina']}")
            logging.info(f"  Trecho: {c['trecho']}\n")
    logging.info("=========================================================================================================") 