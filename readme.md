# Trabalho de Inteligência Artificial

## Descrição
Este projeto é parte da matéria Inteligência Artificial 2024-2 da UFMS. O objetivo é desenvolver um modelo de inteligência artifical que classifique os textos fornecidos, e coletar seus dados de treino utilizando diferentes algoritmos (KNN, Naive Bäyes, MLP Neural Network e Decision Tree) e hiperparâmetros, para então compará-los, afim de entender suas diferentes aplicações e resultados.

## Estrutura do Projeto
- **/pdfs**: Contém os pdf com os textos utilizados para treinar o modelo.
- **main.py**: Executa o script de treino do modelo, desde a separação dos dados até a parte de treinamento e inferência.
- **Arquivos .csv**: Datasheets contendo a acurácia e o F1-Score de cada fold.

## Instalação
Para instalar as dependências do projeto, execute:
```bash
pip install -r requirements.txt
```

## Uso
Para rodar o projeto, utilize o seguinte comando:
```bash
python src/main.py
```

## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.