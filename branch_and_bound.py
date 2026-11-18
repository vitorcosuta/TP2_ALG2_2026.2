import time

class BranchAndBound:
    def __init__(self, dist_matrix, limite_tempo=1800):
        self.dist_matrix = dist_matrix
        self.N = len(dist_matrix)
        self.limite_tempo = limite_tempo
        self.tempo_inicio = None
        
        # Variáveis de estatitisticas
        self.nos_expandidos = 0
        self.nos_podados = 0
        self.melhor_custo = float('inf')
        self.melhor_caminho = []

    def solve(self, warm_custo_inicial_no, warm_caminho_inicial_no):
        self.melhor_custo = warm_custo_inicial_no
        self.melhor_caminho = warm_caminho_inicial_no
        self.tempo_inicio = time.perf_counter()
        
        # alocacao de array de bitmask
        visitado = [False] * self.N
        visitado[0] = True  # nó 0 eh origem origem
        caminho_atual = [0]
        
        self._dfs(no_atual=0, custo_atual=0.0, visitado=visitado, caminho_atual=caminho_atual, profundidade=1)
        
        return self.melhor_custo, self.melhor_caminho, self.nos_expandidos, self.nos_podados

    def _dfs(self, no_atual, custo_atual, visitado, caminho_atual, profundidade):
        # Verificao se passou de 30 min
        if time.perf_counter() - self.tempo_inicio > self.limite_tempo:
            """ deve  ser alterar o retorno"""
            raise TimeoutError("Tempo limite excedido")
            
        self.nos_expandidos += 1

        # Condição de Parada 
        if profundidade == self.N:
            # Soma o custo de retorno à origem
            custo_total = custo_atual + self.dist_matrix[no_atual][0]
            if custo_total < self.melhor_custo:
                self.melhor_custo = custo_total
                self.melhor_caminho = list(caminho_atual) #unica cópia de array necessária
            return

        # Branch
        for proximo_no in range(self.N):
            if not visitado[proximo_no]:
                custo_passo = self.dist_matrix[no_atual][proximo_no]
                novo_custo = custo_atual + custo_passo
                
                # Cálculo do Limite Inferior e Poda
                lb = self._compute_lower_bound(novo_custo, caminho_atual, visitado)
                
                if lb >= self.melhor_custo:
                    self.nos_podados += 1
                    continue # Poda o ramo
                
                # Backtracking
                visitado[proximo_no] = True
                caminho_atual.append(proximo_no)
                
                self._dfs(proximo_no, novo_custo, visitado, caminho_atual, profundidade + 1)
            
                # Reverte o estado para explorar o proximo irmao
                caminho_atual.pop()
                visitado[proximo_no] = False

    def _compute_lower_bound(self, custo_atual, caminho_atual, visitado):
        """Implementar estimativa de custo mínimo dos vértices restantes"""
        pass