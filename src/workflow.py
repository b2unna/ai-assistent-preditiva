from langgraph.graph import StateGraph, START, END
from agents import(
    State,
    agent_detalhe_tecnico,
    agent_perguntas_e_respostas,
    agent_politicas_e_procedimentos,
    agent_tickets,
)
from supervisor import agent_supervisor


def decide_action(state: State):
    #se a resposta já foi gerada pelo supervisor, encerra o fluxo.
    if 'answer' in state:
        return 'end_workflow'
    else:
        #caso contrario, usa a rota para ir para o agente especialista correspondente
        return state['route']
    
def build_workflow():
    workflow = StateGraph(State)

    #adicionar os nós
    workflow.add_node('supervisor_node', agent_supervisor) 
    workflow.add_node('detalhe_tecnico_node', agent_detalhe_tecnico) 
    workflow.add_node('perguntas_e_respostas_node', agent_perguntas_e_respostas) 
    workflow.add_node('politicas_e_procedimentos_node', agent_politicas_e_procedimentos) 
    workflow.add_node('tickets_node', agent_tickets) 

    #adicionar um nó de saída para a resposta direta
    workflow.add_node('end_workflow', lambda x: x)

    #definir o nó inicial
    workflow.add_edge(START, 'supervisor_node')

    #adicione o roteamento condicional
    workflow.add_conditional_edges(
                                  'supervisor_node',
                                  decide_action,
                                  {
                                      "detalhe_tecnico": 'detalhe_tecnico_node',
                                      "perguntas_e_respostas": 'perguntas_e_respostas_node',
                                      "politicas_e_procedimentos": 'politicas_e_procedimentos_node',
                                      "tickets": 'tickets_node',
                                      "end_workflow": END #determina o fluxo se a resposta já foi gerada pelo supervisor.
                                  }                                
 )
     #defina as saídas dos agentes especialistas
    workflow.add_edge('detalhe_tecnico_node', END)
    workflow.add_edge('perguntas_e_respostas_node', END)
    workflow.add_edge('politicas_e_procedimentos_node', END)
    workflow.add_edge('tickets_node', END)
    
    return workflow.compile()

