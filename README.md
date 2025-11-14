## Análise do Controle do Pêndulo Invertido


### Integrantes do grupo
Daniel da Silveira Sahadi

Pablo Carvalho do Nascimentos dos Santos

Thiago Galante Pereira

Willian Nelton Teleken


### a) Dificuldade de Controle Humano

A dificuldade de controlar o sistema como humano depende de alguns fatores principais:

#### **Velocidade do Sistema**
- a velocidade do sistema pode ser determinada pelo numero de tiques do clock, com um numero de ticks menor podemos enviar mais comandos ao sistema e tambem aumentamos o tempo de reação e processamento do jogador humano para a situação atual

#### **Complexidade do Objetivo**
- **Objetivo simples**: Maximizar tempo no ambiente: bastante factível
- **Objetivo complexo**: Manter a posição (x = 0, θ = 0): consideravelmente difícil

- Se o objetivo for controlar o carrinho para mante-lo na posição x = 0 com θ = 0 o o problema se torna muito dificil para um humano, apesar de conseguirmos utilizar noções de fisica para maximizar o tempo do carrinho no ambiente, mante-lo estavel na posição x = 0 e θ = 0 é consideravelmente mais dificil

### b) Como o progresso no aprendizado do controle manual poderia ser estimado?

#### **Problema com Tempo**

- Tentando controlar o carrinho na posição x = 0 e θ = 0 logo notamos que tempo não é uma boa metrica para mensurar essa capacidade. uma vez que conseguimos deixar o carrinho por muito tempo no ambiente e ainda bastante longe desse estado, precisamos de uma metrica melhor

#### **Solução possível: Discretização do Espaço**
- observando o conjunto de valores que o sistema pode assumir notamos que x e θ são limitados no problema, dessa forma podemos efetuar uma discretização dessas variaveis para o espaço de estados e podemos mensurar a performance de aprendizado do ser humano como a capacidade de manter o pendulo no estado mais estavel, considerando como perfomance a frequencia no estado mais estavel. Todavia, isso pode levar a uma politica de tentar encerrar o jogo o mais rapido possivel, pois, ao acabar o jogo rapidamente, o estado inicial (x = 0 e θ = 0) representa uma grande parcela de todos os estados. Assim uma solução multiplicar essa frequencia no estado desejado pelo percentual do tempo de simulação. Com isso, simulações curtas com uma frequencia alta no estado zero são fortemente penalizadas.