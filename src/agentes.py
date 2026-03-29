#importando bibliotecas
from typing import TypedDict, Optional , List #tipando os dados
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage #definindo o formato da mensagem
from config import llm
from retrivers import (
  retriever_manual_tecnico, 
  retriever_perguntas_frequentes,
  retriever_politicas_procedimentos,
  retriever_tickets,
)
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

#Estrutura de mensagens e padrão que está sendo percorrido por detrás dos panos
class State(TypedDict, total=False): #herdando de TypedDict e total=False para permitir chaves opcionais
  query: str
  route : Optional[str] #rota é opcional, pode ser str ou None
  anwser: Optional[str] #resposta é opcional, pode ser str ou None
  chat_history: Optional[List[BaseMessage]] #histórico de mensagens é opcional, pode ser uma lista de mensagens ou None  



def agent_with_retriever(state: State, papel: str, prompt_instructions: str, retriever = None):
    query = state["query"] #recuperar pergunta feita pelo usuário
    chat_history = state.get("chat_history", []) #recuperar histórico de mensagens, se não existir, usar lista vazia
    context = "" #inicializar contexto vazio
   
    if retriever:
        recuperados = retriver.get_relevant_documents(query) #recuperar documentos relevantes usando o retriever
        if recuperados: #se houver documentos recuperados, construir o contexto concatenando o conteúdo das páginas
            context = "\n".join([doc.page_content for doc in recuperados]) #construir o contexto concatenando o conteúdo das páginas recuperadas, separando por nova linha

  #criando estrutura de mensagens. A primeira mensagem é do sistema, definindo o papel do agente e as instruções. Em seguida, adicionamos o histórico de mensagens da conversa, se houver, e por fim, a pergunta atual do usuário junto com o contexto recuperado.
    mensagens = [
        SystemMessage(content=
                      f"Você é um {papel}."
                      f"Suas instruoes:\n{prompt_instructions}\n\n"
                      f" - Use sempre o contexto recuperado para responder a última pergunta do usuário.\n"
                      f" - Use o histórico da conversa para entender o contexto geral e perguntas de acompanhamento. \n"
                      f" - Se nao houver informaçoes relevantes no contexto, diga que nao encontrou dados suficientes para responder a pergunta. \n"
                      f" - Evite inventar informações."
                 ),
                *chat_history, #desempacotar o histórico de mensagens e adicioná-lo à lista de mensagens
          HumanMessage(content=(
               f"Pergunta do usuário: \n{query}\n\n"
               f"Contexto disponivel para esta pergunta: \n{context if context else 'Nenhum contexto disponível.'}"
           )) #adicionar a pergunta atual do usuário como uma mensagem do tipo
    ]
    resposta = llm.invoke(mensagens)#gerar resposta usando o modelo de linguagem, passando a lista de mensagens como entrada
    state["anwser"] = resposta.content #atualizar o estado com a resposta gerada
    return state
 
  
#Criando o agente de detalhe técnico
def agent_detalhe_tecnico(state: State):
    prompt_instructions = (
        "Seja um **especialista em suporte técnico e produto**. " 
        "Você deve responder a perguntas sobre **especificações técnicas**."
        "**instrucoes de instalação**, **manutençao preventiva** e **soluçao de problemas**."
        "Sua resposta deve ser precisa, técnica e objetiva, baseada estritamente no manualk técnico."
        "Para problemas, ofereça uma soluçao clara e passo a passo."
    )
    return agent_with_retriever(
        state, "especialista em detalhes técnicos de produtos", prompt_instructions, retriever_manual_tecnico    )

#Criando o agente de perguntas frequentes
def agent_perguntas_e_respostas(state: State):
    prompt_instructions = (
        "Seja um **especialista em Perguntas e Respostas (FAQ)**. " 
        "Sua função é fornecer respostas diretas e concisas a perguntas comuns."
        "Responda como se estivesse consultando uma base de conhecimento, mantendo a resposta factual e sem rodeios."
        "Se a pergunta se referir a um problema, ofereça a resposta e, se necessário, sugira o contato com o suporte técnico para casos complexos."
    )
    return agent_with_retriever(
        state, "especialista em FAQs", prompt_instructions, retriever_perguntas_frequentes    )

#criando o agente de politicas e procedimentos
def agent_politicas_e_procedimentos(state: State):
    prompt_instructions = (
        "Seja um **especialista em políticas e procedimentos da empresa**. " 
        "Sua tarefa é responder a perguntas sobre **garantia** e **horário de atendimento**."
        "**prazos de SLA** e **regras internas de suporte**."
        "Sua resposta deve ser formal e baseada nos documentos oficiais, farantindo que o cliente entenda as regras e os processos da empresa."
    )
    return agent_with_retriever(
        state, "especialista em políticas e procedimentos da empresa", prompt_instructions, retriever_politicas_procedimentos     )

#criando o agente de tickets de atendimento
def agent_tickets(state: State):
    prompt_instructions = (
        "Seja um **especialista em tickets de atendimento**. " 
        "Você deve fornecer informações precisas sobre **status e detalhes de um chamado existente**."
        "Sua resposta deve ser direta, baseada nos dados do ticket (ticket ID, Status, Responsável, Descrição do produto)."
        "Se o usuário perguntar sobre um ticket específico, forneça as informações correspondentes e relevantes e mantenha a resposta curta e direta."
    )
    return agent_with_retriever(
        state, "especialista em tickets de atendimento", prompt_instructions, retriever_tickets    )