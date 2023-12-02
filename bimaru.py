# bimaru.py: implementação do projeto de Inteligência Artificial 2022/2023.

import sys
from sys import stdin
import copy
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

# constantes
TAMANHO_GRELHA = 10
N_BARCOS_4 = 1
N_BARCOS_3 = 2
N_BARCOS_2 = 3
N_BARCOS_1 = 4

HORIZONTAL = 0
VERTICAL = 1


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        self.valido = True  # se em algum momento o board ficar errado, troca-se isto para false e assim nao se expande mais na procura
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def otimiza_estado(self):
        """aplica todas as açoes certas num dado estado"""

        if self.state_id == 1:
            # definir os barcos que podem ter sido formados pelas dicas
            self.board.procura_barcos_apos_dicas()

        progressos = True
        while progressos == True:
            progressos = False

            # encher linhas e colunas de água ou peças de navio
            if self.encher_linhas_colunas() == True:
                progressos = True
            if self.get_valido() == False:
                return

            # se colocar uma peça de navio entre dois conjuntos de peças de navio forma um navio demasiado grande, colocar água
            for i in range(TAMANHO_GRELHA):
                for j in range(TAMANHO_GRELHA):
                    valor = self.board.get_value(i, j)
                    if valor is None:
                        contador_pecas_1 = 0
                        contador_pecas_2 = 0
                        contador_pecas_3 = 0
                        if (
                            self.board.get_value(i, j - 1) is not None
                            and self.board.get_value(i, j - 1).lower() != "w"
                        ):
                            contador_pecas_1 += 1
                            contador_pecas_2 += 1
                            contador_pecas_3 += 1
                        if (
                            self.board.get_value(i, j - 2) is not None
                            and self.board.get_value(i, j - 2).lower() != "w"
                        ):
                            contador_pecas_1 += 1
                            contador_pecas_3 += 1
                        if (
                            self.board.get_value(i, j - 3) is not None
                            and self.board.get_value(i, j - 3).lower() != "w"
                        ):
                            contador_pecas_1 += 1
                        if (
                            self.board.get_value(i, j + 1) is not None
                            and self.board.get_value(i, j + 1).lower() != "w"
                        ):
                            contador_pecas_1 += 1
                            contador_pecas_2 += 1
                            contador_pecas_3 += 1
                        if (
                            self.board.get_value(i, j + 2) is not None
                            and self.board.get_value(i, j + 2).lower() != "w"
                        ):
                            contador_pecas_2 += 1
                            contador_pecas_3 += 1
                        if (
                            self.board.get_value(i, j + 3) is not None
                            and self.board.get_value(i, j + 3).lower() != "w"
                        ):
                            contador_pecas_2 += 1

                        if (
                            contador_pecas_1 == 4
                            or contador_pecas_2 == 4
                            or contador_pecas_3 == 4
                        ):
                            if self.board.coloca_peça(i, j, "i") == True:
                                progressos = True
                        else:
                            contador_pecas_1 = 0
                            contador_pecas_2 = 0
                            contador_pecas_3 = 0
                            if (
                                self.board.get_value(i - 1, j) is not None
                                and self.board.get_value(i - 1, j).lower() != "w"
                            ):
                                contador_pecas_1 += 1
                                contador_pecas_2 += 1
                                contador_pecas_3 += 1
                            if (
                                self.board.get_value(i - 2, j) is not None
                                and self.board.get_value(i - 2, j).lower() != "w"
                            ):
                                contador_pecas_1 += 1
                                contador_pecas_3 += 1
                            if (
                                self.board.get_value(i - 3, j) is not None
                                and self.board.get_value(i - 3, j).lower() != "w"
                            ):
                                contador_pecas_1 += 1
                            if (
                                self.board.get_value(i + 1, j) is not None
                                and self.board.get_value(i + 1, j).lower() != "w"
                            ):
                                contador_pecas_1 += 1
                                contador_pecas_2 += 1
                                contador_pecas_3 += 1
                            if (
                                self.board.get_value(i + 2, j) is not None
                                and self.board.get_value(i + 2, j).lower() != "w"
                            ):
                                contador_pecas_2 += 1
                                contador_pecas_3 += 1
                            if (
                                self.board.get_value(i + 3, j) is not None
                                and self.board.get_value(i + 3, j).lower() != "w"
                            ):
                                contador_pecas_2 += 1
                            if (
                                contador_pecas_1 == 4
                                or contador_pecas_2 == 4
                                or contador_pecas_3 == 4
                            ):
                                if self.board.coloca_peça(i, j, "i") == True:
                                    progressos = True

                    # verificar se o board ficou errado com alguma alteração
                    if self.board.get_board_errado() == True:
                        self.set_invalido()
                        return

            # definir navios -- rafa

            if self.board.get_n_barcos_restantes(4) > 0:
                if self.definir_navios_coluna(4) == True:
                    progressos = True
                if self.definir_navios_linha(4) == True:
                    progressos = True
            elif self.board.get_n_barcos_restantes(3) > 0:
                if self.definir_navios_coluna(3) == True:
                    progressos = True
                if self.definir_navios_linha(3) == True:
                    progressos = True
            elif self.board.get_n_barcos_restantes(2) > 0:
                if self.definir_navios_coluna(2) == True:
                    progressos = True
                if self.definir_navios_linha(2) == True:
                    progressos = True
            elif self.board.get_n_barcos_restantes(1) > 0:
                if self.definir_navios_linha(1) == True:
                    progressos = True



    def definir_navios_linha(self, tamanho_barco):
        """define navios do maior tamanho disponível nas linhas todas"""
        progressos = False
        for i in range(TAMANHO_GRELHA):
            if self.board.valores_iniciais_linhas[i] >= tamanho_barco:
                spots = []
                for k in range(tamanho_barco):  # primeiras n posicoes da linha
                    spots.append(self.board.get_value(i, k))
                for j in range(tamanho_barco, TAMANHO_GRELHA + 1):
                    if (
                        ("i" in spots)
                        and (  # primeira peça
                            spots[0] is not None and spots[0].lower() in ("i", "l")
                        )
                        and all(  # peças do meio do barco
                            item is not None and item.lower() in ("i", "m")
                            for item in spots[1:-1]
                        )
                        and (  # ultima peça
                            spots[-1] is not None and spots[-1].lower() in ("i", "r")
                        )
                    ):
                        progressos = True
                        if tamanho_barco > 1:
                            for l in range(tamanho_barco):
                                if spots[l] == "i":
                                    if l == 0:
                                        spots[l] = "l"
                                    elif l == tamanho_barco - 1:
                                        spots[l] = "r"
                                    else:
                                        spots[l] = "m"
                                if (
                                    self.board.coloca_peça(
                                        i, j - tamanho_barco + l, spots[l]
                                    )
                                    != True
                                ):
                                    if self.board.get_board_errado() == True:
                                        self.set_invalido()
                                        return False
                        else:
                            if self.board.coloca_peça(i, j - 1, "c") != True:
                                if self.board.get_board_errado() == True:
                                    self.set_invalido()
                                    return False
                        self.board.set_n_barcos_restantes(tamanho_barco)
                        if self.board.get_n_barcos_restantes(tamanho_barco) == 0:
                            return True
                    # trocar os valores da lista spots, para os valores das colunas imediatamente a seguir
                    for l in range(tamanho_barco - 1):
                        spots[l] = spots[l + 1]
                    spots[tamanho_barco - 1] = self.board.get_value(i, j)
        return progressos

    def definir_navios_coluna(self, tamanho_barco):
        """define navios do maior tamanho disponível nas colunas todas"""
        progressos = False
        for i in range(TAMANHO_GRELHA):
            if self.board.valores_iniciais_colunas[i] >= tamanho_barco:
                spots = []
                for k in range(tamanho_barco):  # primeiras n posicoes da linha
                    spots.append(self.board.get_value(k, i))
                for j in range(tamanho_barco, TAMANHO_GRELHA + 1):
                    if (
                        ("i" in spots)
                        and (  # primeira peça
                            spots[0] is not None and spots[0].lower() in ("i", "t")
                        )
                        and all(  # peças do meio do barco
                            item is not None and item.lower() in ("i", "m")
                            for item in spots[1:-1]
                        )
                        and (  # ultima peça
                            spots[-1] is not None and spots[-1].lower() in ("i", "b")
                        )
                    ):
                        progressos = True

                        for l in range(tamanho_barco):
                            if spots[l] == "i":
                                if l == 0:
                                    spots[l] = "t"
                                elif l == tamanho_barco - 1:
                                    spots[l] = "b"
                                else:
                                    spots[l] = "m"
                            if (
                                self.board.coloca_peça(
                                    j - tamanho_barco + l, i, spots[l]
                                )
                                != True
                            ):
                                if self.board.get_board_errado() == True:
                                    self.set_invalido()
                                    return False
                        self.board.set_n_barcos_restantes(tamanho_barco)
                        if self.board.get_n_barcos_restantes(tamanho_barco) == 0:
                            return True
                    # trocar os valores da lista spots, para os valores das colunas imediatamente a seguir
                    for l in range(tamanho_barco - 1):
                        spots[l] = spots[l + 1]
                    spots[tamanho_barco - 1] = self.board.get_value(j, i)
        return progressos

    def encher_linhas_colunas(self):
        """função que enche linhas e colunas de água ou peças de navio"""

        progressos = False
        for i in range(TAMANHO_GRELHA):
            # encher linhas de água ou peças de navio
            if self.board.get_navios_linha(i) == 0 and self.board.get_agua_linha(i) > 0:
                if self.board.encher_linha_agua(i) == True:
                    progressos = True
                if self.board.get_board_errado() == True:
                    self.set_invalido()
                    return False
            elif (
                self.board.get_agua_linha(i) == 0 and self.board.get_navios_linha(i) > 0
            ):
                if self.board.encher_linha_navio(i) == True:
                    progressos = True
                if self.board.get_board_errado() == True:
                    self.set_invalido()
                    return False

            # encher colunas de água ou peças de navio
            if (
                self.board.get_navios_coluna(i) == 0
                and self.board.get_agua_coluna(i) > 0
            ):
                if self.board.encher_col_agua(i) == True:
                    progressos = True
                if self.board.get_board_errado() == True:
                    self.set_invalido()
                    return False
            elif (
                self.board.get_agua_coluna(i) == 0
                and self.board.get_navios_coluna(i) > 0
            ):
                if self.board.encher_col_navio(i) == True:
                    progressos = True
                if self.board.get_board_errado() == True:
                    self.set_invalido()
                    return False

            for j in range(TAMANHO_GRELHA):
                if self.board.get_value(i, j) == "M":
                    if self.board.circundar_navio(i, j) != True:
                        if self.board.get_board_errado() == True:
                            self.set_invalido()
                            return False
                    else:
                        progressos = True

        return progressos

    def get_valido(self):
        """Retorna True se o estado é válido/correto, False caso contrário"""
        return self.valido

    def set_invalido(self):
        """Torna o estado inválido/errado, para parar a procura neste ramo"""
        self.valido = False


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(
        self, board, valores_linhas, valores_colunas, barco_4, barco_3, barco_2, barco_1
    ):
        self.board = board  # array 2d que simula a grelha
        self.valores_linhas = valores_linhas  # nº de peças de barco de cada linha
        self.valores_colunas = valores_colunas  # nº de peças de barco de cada coluna
        self.valores_linhas_agua = [
            10 - num for num in valores_linhas
        ]  # nº de peças de água em cada linha
        self.valores_colunas_agua = [
            10 - num for num in valores_colunas
        ]  # nº de peças de água em cada coluna
        self.valores_iniciais_linhas = tuple(
            valores_linhas
        )  # nº de peças de barco iniciais de cada linha
        self.valores_iniciais_colunas = tuple(
            valores_colunas
        )  # nº de peças de barco iniciais de cada coluna
        self.barco_4 = barco_4  # nº de barcos de tamanho 4
        self.barco_3 = barco_3  # nº de barcos de tamanho 3
        self.barco_2 = barco_2  # nº de barcos de tamanho 2
        self.barco_1 = barco_1  # nº de barcos de tamanho 1
        self.board_errado = False

    def get_value(self, row: int, col: int):
        """Devolve o valor na respetiva posição do tabuleiro.
        letra que está nessa posição, None se a posicao estiver vazia,
        X se for fora dos limites da grelha
        """
        if row < 0 or col < 0 or row >= TAMANHO_GRELHA or col >= TAMANHO_GRELHA:
            return None
        value = self.board[row][col]
        if value == "":
            return None
        return value

    def adjacent_vertical_values(self, row: int, col: int):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (self.get_value(row - 1, col), self.get_value(row + 1, col))

    def adjacent_horizontal_values(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_value(row, col - 1), self.get_value(row, col + 1))

    def adjacent_diagonal_values(self, row: int, col: int):
        """Devolve os valores nas diagonais"""
        return (
            self.get_value(row - 1, col - 1),
            self.get_value(row - 1, col + 1),
            self.get_value(row + 1, col - 1),
            self.get_value(row + 1, col + 1),
        )

    def get_n_barcos_restantes(self, tamanho: int):
        """
        Devolve o número de barcos restantes de um determinado tamanho
        """
        if tamanho == 1:
            return self.barco_1
        elif tamanho == 2:
            return self.barco_2
        elif tamanho == 3:
            return self.barco_3
        elif tamanho == 4:
            return self.barco_4
        return -1  # nunca chega aqui

    def set_n_barcos_restantes(self, tamanho: int):
        """
        diminui o nº de barcos restantes de um determinado tamanho
        """
        if tamanho == 1:
            self.barco_1 -= 1
        elif tamanho == 2:
            self.barco_2 -= 1
        elif tamanho == 3:
            self.barco_3 -= 1
        elif tamanho == 4:
            self.barco_4 -= 1

    def get_board_errado(self):
        """devolve True se o board está errado"""
        return self.board_errado

    def set_board_errado(self):
        """coloca o valor de board_errado a True, quando há uma ação que torna o board errado"""
        self.board_errado = True

    def get_navios_linha(self, linha):
        return self.valores_linhas[linha]

    def get_navios_coluna(self, coluna):
        return self.valores_colunas[coluna]

    def set_navios_linha(self, linha, n_partes):
        self.valores_linhas[linha] -= n_partes
        if self.get_navios_linha(linha) < 0:
            self.set_board_errado()

    def set_navios_coluna(self, coluna, n_partes):
        self.valores_colunas[coluna] -= n_partes
        if self.get_navios_coluna(coluna) < 0:
            self.set_board_errado()

    def get_agua_linha(self, linha):
        return self.valores_linhas_agua[linha]

    def get_agua_coluna(self, coluna):
        return self.valores_colunas_agua[coluna]

    def set_agua_linha(self, linha, n_partes):
        self.valores_linhas_agua[linha] -= n_partes
        if self.get_agua_linha(linha) < 0:
            self.set_board_errado()

    def set_agua_coluna(self, coluna, n_partes):
        self.valores_colunas_agua[coluna] -= n_partes
        if self.get_agua_coluna(coluna) < 0:
            self.set_board_errado()

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """

        linhas = input().strip().split("\t")  # 1ª linha do input
        if linhas[0] == "ROW":
            linhas = linhas[1:]  # lista com os nºs das linhas
            linhas = [int(num) for num in linhas]  # passar para int

        colunas = input().strip().split("\t")  # igual para as colunas
        if colunas[0] == "COLUMN":
            colunas = colunas[1:]
            colunas = [int(num) for num in colunas]

        board = np.empty((10, 10), dtype=str)
        n_dicas = int(input().strip())  # ler a linha do nº de dicas
        for i in range(n_dicas):
            dica = input().strip().split("\t")  # ler cada dica
            if dica[0] == "HINT":
                board[int(dica[1])][int(dica[2])] = dica[3]

        return Board(
            board, linhas, colunas, N_BARCOS_4, N_BARCOS_3, N_BARCOS_2, N_BARCOS_1
        )

    def coloca_peça(self, linha: int, coluna: int, valor: str):
        """colocar peça de navio (definida ou indefinida)"""
        progresso = False
        if linha < 0 or linha > 9 or coluna < 0 or coluna > 9:
            return progresso
        if self.get_value(linha, coluna) is None:
            self.board[linha][coluna] = valor
            valor = valor.lower()
            if valor == "w":
                # agua
                progresso = True
                self.set_agua_linha(linha, 1)
                self.set_agua_coluna(coluna, 1)
                if (
                    # se ja ha aguas a mais nessa linha / coluna
                    self.get_agua_linha(linha) < 0
                    or self.get_agua_coluna(coluna) < 0
                ):
                    self.set_board_errado()
                    return False
            else:
                # parte de navio
                self.set_navios_linha(linha, 1)
                self.set_navios_coluna(coluna, 1)
                if self.circundar_navio(linha, coluna) != True:
                    if self.get_board_errado() == True:
                        return False
                if (
                    self.get_navios_linha(linha) < 0
                    or self.get_navios_coluna(coluna) < 0
                ):
                    # se ja ha partes de navio a mais nessa linha / coluna
                    self.set_board_errado()
                    return False
        elif self.get_value(linha, coluna) == "i" and valor != "w" and valor != "i":
            # se a letra nessa posicao for 'i' mas a que se ta a tentar meter é uma parte de barco definida
            self.board[linha][coluna] = valor
            if self.circundar_navio(linha, coluna) != True:
                if self.get_board_errado() == True:
                    return False
        elif valor.lower() == "w" and self.get_value(linha, coluna).lower() != "w":
            # se se tentar colocar água e ja houver uma peça de barco, o board fica errado
            self.set_board_errado()
        elif self.get_value(linha, coluna).lower() == "w" and valor.lower() != "w":
            # se se tentar colocar uma peça de barco e já houver uma água, o board fica errado
            self.set_board_errado()
        elif valor.lower() not in ("w", "i") and self.get_value(
            linha, coluna
        ).lower() not in ("w", "i", valor.lower()):
            # se se tentar colocar uma peça definida e ja la ha outra peça definida diferente
            self.set_board_errado()

        return progresso

    def encher_linha_agua(self, linha: int):
        """encher linha de água"""
        progresso = False
        for i in range(TAMANHO_GRELHA):
            if self.get_value(linha, i) is None:
                if self.coloca_peça(linha, i, "w") != True:
                    return False
                if self.get_board_errado() == True:
                    return progresso
                progresso = True
        return progresso

    def encher_col_agua(self, col: int):
        """encher coluna de água"""
        progresso = False
        for i in range(TAMANHO_GRELHA):
            if self.get_value(i, col) is None:
                if self.coloca_peça(i, col, "w") != True:
                    return False
                if self.get_board_errado() == True:
                    return progresso
                progresso = True
        return progresso

    def encher_linha_navio(self, linha: int):
        """encher linha com partes de barco"""
        progresso = False
        for i in range(TAMANHO_GRELHA):
            if self.get_value(linha, i) is None:
                if self.coloca_peça(linha, i, "i") != True:
                    return False
                if self.get_board_errado() == True:
                    return progresso
                progresso = True
        return progresso

    def encher_col_navio(self, coluna: int):
        """encher coluna com partes de barco"""
        progresso = False
        for i in range(TAMANHO_GRELHA):
            if self.get_value(i, coluna) is None:
                if self.coloca_peça(i, coluna, "i") != True:
                    return False
                if self.get_board_errado() == True:
                    return progresso
                progresso = True
        return progresso

    def circundar_navio(self, row: int, col: int):
        """colocar água, e outras partes de navio sabidas, à volta de uma parte de navio"""
        progresso = False
        # TODO verificar se alguma destas modificacoes torna o board errado

        alvo = self.get_value(row, col)
        if alvo is not None and alvo.lower() != "w":
            # peça de navio
            alvo = alvo.lower()

            # colocar água nas posições diagonais
            if self.coloca_peça(row - 1, col - 1, "w") == True:
                progresso = True
            if self.coloca_peça(row + 1, col - 1, "w") == True:
                progresso = True
            if self.coloca_peça(row - 1, col + 1, "w") == True:
                progresso = True
            if self.coloca_peça(row + 1, col + 1, "w") == True:
                progresso = True
            if self.get_board_errado() == True:
                return False

            # top ou bottom
            if alvo == "t" or alvo == "b":
                # colocar agua à esquerda e direita
                if self.coloca_peça(row, col - 1, "w") == True:
                    progresso = True
                if self.coloca_peça(row, col + 1, "w") == True:
                    progresso = True
                if alvo == "t":
                    # colocar agua em cima, e parte de navio indefinida em baixo
                    if self.coloca_peça(row - 1, col, "w") == True:
                        progresso = True
                    if self.coloca_peça(row + 1, col, "i") == True:
                        progresso = True
                elif alvo == "b":
                    # colocar agua em baixo, e parte de navio indefinida em cima
                    if self.coloca_peça(row + 1, col, "w") == True:
                        progresso = True
                    if self.coloca_peça(row - 1, col, "i") == True:
                        progresso = True
                if self.get_board_errado() == True:
                    return False

            # left ou right
            elif alvo == "l" or alvo == "r":
                # colocar agua em cima e em baixo
                if self.coloca_peça(row - 1, col, "w") == True:
                    progresso = True
                if self.coloca_peça(row + 1, col, "w") == True:
                    progresso = True
                if alvo == "l":
                    # colocar agua à esquerda e parte de navio indefinida à direita
                    if self.coloca_peça(row, col - 1, "w") == True:
                        progresso = True
                    if self.coloca_peça(row, col + 1, "i") == True:
                        progresso = True
                elif alvo == "r":
                    # colocar agua à direita e parte de navio indefinida à esquerda
                    if self.coloca_peça(row, col + 1, "w") == True:
                        progresso = True
                    if self.coloca_peça(row, col - 1, "i") == True:
                        progresso = True
                if self.get_board_errado() == True:
                    return False

            # circulo
            elif alvo == "c":
                # água tudo à volta
                if self.coloca_peça(row - 1, col, "w") == True:
                    progresso = True
                if self.coloca_peça(row + 1, col, "w") == True:
                    progresso = True
                if self.coloca_peça(row, col - 1, "w") == True:
                    progresso = True
                if self.coloca_peça(row, col + 1, "w") == True:
                    progresso = True
                if self.get_board_errado() == True:
                    return False

            # meio
            elif alvo == "m":
                # limites da grelha ou nº limitado de peças de barco
                if col == 0 or col == 9 or self.valores_iniciais_linhas[row] < 3:
                    if self.coloca_peça(row - 1, col, "i") == True:
                        progresso = True
                    if self.coloca_peça(row + 1, col, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso
                elif row == 0 or row == 9 or self.valores_iniciais_colunas[col] < 3:
                    if self.coloca_peça(row, col - 1, "i") == True:
                        progresso = True
                    if self.coloca_peça(row, col + 1, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso

                # se ja houver alguma peça de barco adjacente
                elif (
                    self.get_value(row - 1, col) is not None
                    and self.get_value(row - 1, col).lower() != "w"
                ):
                    if self.coloca_peça(row + 1, col, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso
                elif (
                    self.get_value(row + 1, col) is not None
                    and self.get_value(row + 1, col).lower() != "w"
                ):
                    if self.coloca_peça(row - 1, col, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso
                elif (
                    self.get_value(row, col - 1) is not None
                    and self.get_value(row, col - 1).lower() != "w"
                ):
                    if self.coloca_peça(row, col + 1, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso
                elif (
                    self.get_value(row, col + 1) is not None
                    and self.get_value(row, col + 1).lower() != "w"
                ):
                    if self.coloca_peça(row, col - 1, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso

                # se ja houver demasiados barcos e as posicoes adjacentes ainda estao vazias
                elif (
                    self.adjacent_horizontal_values(row, col) == (None, None)
                    and self.get_navios_linha(row) < 2
                ):
                    if self.coloca_peça(row - 1, col, "i") == True:
                        progresso = True
                    if self.coloca_peça(row + 1, col, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso
                elif (
                    self.adjacent_vertical_values(row, col) == (None, None)
                    and self.valores_colunas[col] < 2
                ):
                    if self.coloca_peça(row, col - 1, "i") == True:
                        progresso = True
                    if self.coloca_peça(row, col + 1, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso

                # se ja houver agua numa das posicoes adjacentes
                cima = self.get_value(row - 1, col)
                baixo = self.get_value(row + 1, col)
                if self.adjacent_horizontal_values(row, col) == (None, None) and (
                    (cima is not None and cima.lower() == "w")
                    or (baixo is not None and baixo.lower() == "w")
                ):
                    if self.coloca_peça(row, col - 1, "i") == True:
                        progresso = True
                    if self.coloca_peça(row, col + 1, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso

                esquerda = self.get_value(row, col - 1)
                direita = self.get_value(row, col + 1)
                if self.adjacent_vertical_values(row, col) == (None, None) and (
                    (esquerda is not None and esquerda.lower() == "w")
                    or (direita is not None and direita.lower() == "w")
                ):
                    if self.coloca_peça(row - 1, col, "i") == True:
                        progresso = True
                    if self.coloca_peça(row + 1, col, "i") == True:
                        progresso = True
                    if self.get_board_errado() == True:
                        return False
                    return progresso

        return progresso

    def print_board(self):
        """imprime o navio no formato de output indicado"""
        for i in range(TAMANHO_GRELHA):
            for j in range(TAMANHO_GRELHA):
                if self.board[i][j] == "w":
                    print(".", end="")
                else:
                    print(self.board[i][j], end="")
            print("")

    def procura_barcos_apos_dicas(self):
        """verifica se foram criados barcos completos apos colocaçao das dicas"""
        for i in range(TAMANHO_GRELHA):
            for j in range(TAMANHO_GRELHA):
                # pode ser util verificar se ainda n foram colocados barcos ou se o valor da linha/coluna é 0
                valor = self.get_value(i, j)
                if valor is not None:
                    if valor == "W":
                        self.set_agua_linha(i, 1)
                        self.set_agua_coluna(j, 1)
                    elif valor in ("T", "B", "R", "L", "M", "C"):
                        self.set_navios_linha(i, 1)
                        self.set_navios_coluna(j, 1)
                if self.circundar_navio(i, j) or True:
                    pass
                if valor == "C":
                    self.set_n_barcos_restantes(1)
                elif valor == "L":
                    next_1 = self.get_value(i, j + 1)
                    next_2 = self.get_value(i, j + 2)
                    next_3 = self.get_value(i, j + 3)
                    if next_1 == "R":  # barco de 2
                        self.set_n_barcos_restantes(2)
                    elif (
                        next_1 == "i" or next_1 == "M"
                    ) and next_2 == "R":  # barco de 3
                        if self.coloca_peça(i, j + 1, "m") or True:
                            self.set_n_barcos_restantes(3)
                    elif (
                        (next_1 == "i" or next_1 == "M")
                        and (next_2 == "i" or next_2 == "M")
                        and next_3 == "R"
                    ):  # barco de 4
                        if (
                            self.coloca_peça(i, j + 1, "m")
                            or self.coloca_peça(i, j + 2, "m")
                            or True
                        ):
                            self.set_n_barcos_restantes(4)

                elif valor == "T":
                    next_1 = self.get_value(i + 1, j)
                    next_2 = self.get_value(i + 2, j)
                    next_3 = self.get_value(i + 3, j)
                    if next_1 == "B":  # barco de 2
                        self.set_n_barcos_restantes(2)
                    elif (
                        next_1 == "i" or next_1 == "M"
                    ) and next_2 == "B":  # barco de 3
                        if self.coloca_peça(i + 1, j, "m") or True:
                            self.set_n_barcos_restantes(3)
                    elif (
                        (next_1 == "i" or next_1 == "M")
                        and (next_2 == "i" or next_2 == "M")
                        and next_3 == "B"
                    ):  # barco de 4
                        if (
                            self.coloca_peça(i + 1, j, "m")
                            or self.coloca_peça(i + 2, j, "m")
                            or True
                        ):
                            self.set_n_barcos_restantes(4)


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)
        # TODO
        pass

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #   (direção, linha, coluna, tamanho do barco)

        # se o nº de barcos de algum tamanho for menor que 0, ha barcos a mais
        # e o estado está errado, nao vale a pena aprofundar mais
        actions = []
        if (
            state.board.get_n_barcos_restantes(4) < 0
            or state.board.get_n_barcos_restantes(3) < 0
            or state.board.get_n_barcos_restantes(2) < 0
            or state.board.get_n_barcos_restantes(1) < 0
            or state.get_valido() == False
        ):
            return actions

        # posicoes possiveis de colocar um dos maiores barcos possíveis
        if state.board.get_n_barcos_restantes(4) > 0:
            actions = self.get_barcos_possiveis_linha(
                state, 4
            ) + self.get_barcos_possiveis_coluna(state, 4)

        elif state.board.get_n_barcos_restantes(3) > 0:
            actions = self.get_barcos_possiveis_linha(
                state, 3
            ) + self.get_barcos_possiveis_coluna(state, 3)

        elif state.board.get_n_barcos_restantes(2) > 0:
            actions = self.get_barcos_possiveis_linha(
                state, 2
            ) + self.get_barcos_possiveis_coluna(state, 2)

        elif state.board.get_n_barcos_restantes(1) > 0:
            actions = self.get_barcos_possiveis_linha(
                state, 1
            ) + self.get_barcos_possiveis_coluna(state, 1)

        return actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        #   (direção, linha, coluna, tamanho do barco)

        estado_resultado = BimaruState(copy.deepcopy(state.board))

        if action[0] == HORIZONTAL:
            linha = action[1]
            coluna = action[2]
            tamanho_barco = action[3]

            # 1ª peça
            if estado_resultado.board.coloca_peça(linha, coluna, "l") == True:
                pass
            # peças do meio
            for i in range(1, tamanho_barco - 1):
                if estado_resultado.board.coloca_peça(linha, coluna + i, "m") == True:
                    pass
            # última peça
            if (
                estado_resultado.board.coloca_peça(
                    linha, coluna + tamanho_barco - 1, "r"
                )
                == True
            ):
                pass

            if estado_resultado.board.get_board_errado() == True:
                estado_resultado.set_invalido()

            estado_resultado.board.set_n_barcos_restantes(tamanho_barco)

        elif action[0] == VERTICAL:
            linha = action[1]
            coluna = action[2]
            tamanho_barco = action[3]

            # 1ª peça
            if estado_resultado.board.coloca_peça(linha, coluna, "t") == True:
                pass
            # peças do meio
            for i in range(1, tamanho_barco - 1):
                if estado_resultado.board.coloca_peça(linha + i, coluna, "m") == True:
                    pass
            # última peça
            if (
                estado_resultado.board.coloca_peça(
                    linha + tamanho_barco - 1, coluna, "b"
                )
                == True
            ):
                pass

            if estado_resultado.board.get_board_errado() == True:
                estado_resultado.set_invalido()

            estado_resultado.board.set_n_barcos_restantes(tamanho_barco)

        estado_resultado.otimiza_estado()
        return estado_resultado

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        if (
            state.board.get_n_barcos_restantes(4) == 0
            and state.board.get_n_barcos_restantes(3) == 0
            and state.board.get_n_barcos_restantes(2) == 0
            and state.board.get_n_barcos_restantes(1) == 0
            and state.board.valores_linhas == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            and state.board.valores_colunas == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            and state.get_valido() == True
        ):
            return True
        return False

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A* se necessária para melhorar o desempenho."""
        # TODO
        pass

    # TODO: outros metodos da classe

    def get_barcos_possiveis_linha(self, state: BimaruState, tamanho_barco):
        """
        retorna uma lista de todas as posicoes onde se pode colocar um barco do tamanho passado como argumento, NA HORIZONTAL
        """
        actions = []
        for i in range(TAMANHO_GRELHA):
            if state.board.valores_iniciais_linhas[i] >= tamanho_barco:
                spots = []
                # peças de navio que a linha ja tem
                peças_navio_linha = state.board.get_navios_linha(i)
                for k in range(tamanho_barco):
                    spots.append(state.board.get_value(i, k))
                for j in range(tamanho_barco, TAMANHO_GRELHA + 1):
                    var_contr = 0  # variavel de controlo
                    for k in range(j - tamanho_barco, j):
                        if state.board.get_value(i, k) is None:
                            if state.board.get_navios_coluna(k) >= 1:
                                var_contr = 1
                        if state.board.get_value(i, k) == "w":
                            var_contr = 0  # se for zero é porque pelo menos um dos spots é agua por isso termina o for
                            break
                    if (
                        var_contr == 1
                        and peças_navio_linha - spots.count(None) >= 0
                        and self.posicoes_barco_validas(spots, HORIZONTAL) == True
                        and (  # verifica se os limites do barco sao agua ou espaço vazio
                            state.board.get_value(i, j) is None
                            or state.board.get_value(i, j).lower() == "w"
                        )
                        and (
                            state.board.get_value(i, j - (tamanho_barco + 1)) is None
                            or state.board.get_value(i, j - (tamanho_barco + 1)).lower()
                            == "w"
                        )
                    ):
                        actions.append(
                            (HORIZONTAL, i, j - tamanho_barco, tamanho_barco)
                        )
                    for l in range(tamanho_barco - 1):
                        spots[l] = spots[l + 1]
                    spots[tamanho_barco - 1] = state.board.get_value(i, j)

        # print("actions da linha", actions)
        return actions

    def get_barcos_possiveis_coluna(self, state: BimaruState, tamanho_barco):
        """
        retorna uma lista de todas as posicoes onde se pode colocar um barco do tamanho passado como argumento, NA VERTICAL
        """
        actions = []
        for i in range(TAMANHO_GRELHA):
            if state.board.valores_iniciais_colunas[i] >= tamanho_barco:
                spots = []
                peças_navio_coluna = state.board.get_navios_coluna(i)
                for k in range(tamanho_barco):
                    spots.append(state.board.get_value(k, i))
                for j in range(tamanho_barco, TAMANHO_GRELHA + 1):
                    var_contr = 0
                    for k in range(j - tamanho_barco, j):
                        if state.board.get_value(k, i) is None:
                            if state.board.get_navios_linha(k) >= 1:
                                var_contr = 1
                        if state.board.get_value(k, i) == "w":
                            var_contr = 0
                            break
                    if (
                        var_contr == 1
                        and peças_navio_coluna - spots.count(None) >= 0
                        and self.posicoes_barco_validas(spots, VERTICAL) == True
                        and (  # verifica se os limites do barco sao agua ou espaço vazio
                            state.board.get_value(j, i) is None
                            or state.board.get_value(j, i).lower() == "w"
                        )
                        and (
                            state.board.get_value(j - (tamanho_barco + 1), i) is None
                            or state.board.get_value(j - (tamanho_barco + 1), i).lower()
                            == "w"
                        )
                    ):
                        actions.append((VERTICAL, j - tamanho_barco, i, tamanho_barco))
                    for l in range(tamanho_barco - 1):
                        spots[l] = spots[l + 1]
                    spots[tamanho_barco - 1] = state.board.get_value(j, i)
        # print("actions da coluna", actions)
        return actions

    def posicoes_barco_validas(self, spots, direcao):
        """verifica se um dado conjunto de n posicoes é válido para colocar um barco"""
        if direcao == HORIZONTAL:
            if (
                (
                    (spots[0] is None or spots[0].lower() in ("l", "i"))
                    and spots[0] != "w"
                )
                and (
                    (spots[-1] is None or spots[-1].lower() in ("r", "i"))
                    and spots[-1] != "w"
                )
                and all(
                    (item != "w" and (item is None or item.lower() in ("i", "m")))
                    for item in spots[1:-1]
                )
            ):
                return True
            else:
                return False
        elif direcao == VERTICAL:
            if (
                (
                    (spots[0] is None or spots[0].lower() in ("t", "i"))
                    and spots[0] != "w"
                )
                and (
                    (spots[-1] is None or spots[-1].lower() in ("b", "i"))
                    and spots[-1] != "w"
                )
                and all(
                    (item != "w" and (item is None or item.lower() in ("i", "m")))
                    for item in spots[1:-1]
                )
            ):
                return True
            else:
                return False

        return False  # nunca chega até aqui


if __name__ == "__main__":
    # Ler o ficheiro do standard input,
    board = Board.parse_instance()
    problem = Bimaru(board)
    problem.initial.otimiza_estado()

    # Usar uma técnica de procura para resolver a instância,
    goal_node = depth_first_tree_search(problem)
    # Retirar a solução a partir do nó resultante,
    print("Is goal?", problem.goal_test(goal_node.state))
    # Imprimir para o standard output no formato indicado.
    goal_node.state.board.print_board()
