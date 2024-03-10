data_header = [
    'Report.Timestamp',
    'Session.Trial.UID',
    'Session.Block.UID',
    'Session.Block.Trial.UID',
    'Session.Block.ID',
    'Session.Block.Trial.ID',
    'Session.Block.Name',

    'Trial.ID',
    'Cycle.ID',
    'Name',
    'Relation',
    'Comparisons',

    'Result',
    'CounterHit',
    'CounterHit.MaxConsecutives',
    'CounterMiss',
    'CounterMiss.MaxConsecutives',
    'CounterNone',
    'CounterNone.MaxConsecutives',
    'Sample-Position.1',
    'Comparison-Position.1',
    'Comparison-Position.2',
    'Comparison-Position.3',
    'Response',

    'HasDifferentialReinforcement',
    'Latency']

timestamps_header = [
    'Timestamp',
    'Session.Trial.UID',
    'Session.Block.UID',
    'Event',
    'Event.Annotation']

session_name_dict = {
    'Pre-training-0' : 'Ciclo1-0-Pre-treino.csv',
    'Probes-CD-12-training-8-generalization-words-1' : 'Ciclo1-7-Sondas-CD-Palavras-12-ensino-8-generalizacao.csv',
    'Training-AB-1' : 'Ciclo1-1-Treino-AB.csv',
    'Training-AC-1' : 'Ciclo1-2a-Treino-AC-CD.csv',
    'Training-AC-cumulative-1' : 'Ciclo1-2b-Treino-AC-Ref-Intermitente.csv',
    'Probes-BC-CB-training-words-1' : 'Ciclo1-3-Sondas-BC-CB-Palavras-de-ensino.csv',
    'Probes-BC-CB-constant-words-1' : 'Ciclo1-4-Sondas-BC-CB-Palavras-reservadas.csv',
    'Probes-CD-generalization-and-constant-words-1' : 'Ciclo1-5-Sondas-CD-Palavras-generalizacao-reservadas.csv',
    'Probes-AC-generalization-and-constant-words-1' : 'Ciclo1-6-Sondas-AC-Palavras-generalizacao-reservadas.csv',

    'Probes-CD-12-training-8-generalization-words-2' : 'Ciclo2-7-Sondas-CD-Palavras-12-ensino-8-generalizacao.csv',
    'Training-AB-2' : 'Ciclo2-1-Treino-AB.csv',
    'Training-AC-2' : 'Ciclo2-2a-Treino-AC-CD.csv',
    'Training-AC-cumulative-2' : 'Ciclo2-2b-Treino-AC-Ref-Intermitente.csv',
    'Probes-BC-CB-training-words-2' : 'Ciclo2-3-Sondas-BC-CB-Palavras-de-ensino.csv',
    'Probes-BC-CB-constant-words-2' : 'Ciclo2-4-Sondas-BC-CB-Palavras-reservadas.csv',
    'Probes-CD-generalization-and-constant-words-2' : 'Ciclo2-5-Sondas-CD-Palavras-generalizacao-reservadas.csv',
    'Probes-AC-generalization-and-constant-words-2' : 'Ciclo2-6-Sondas-AC-Palavras-generalizacao-reservadas.csv',

    'Probes-CD-12-training-8-generalization-words-3' : 'Ciclo3-7-Sondas-CD-Palavras-12-ensino-8-generalizacao.csv',
    'Training-AB-3' : 'Ciclo3-1-Treino-AB.csv',
    'Training-AC-3' : 'Ciclo3-2a-Treino-AC-CD.csv',
    'Training-AC-cumulative-3' : 'Ciclo3-2b-Treino-AC-Ref-Intermitente.csv',
    'Probes-BC-CB-training-words-3' : 'Ciclo3-3-Sondas-BC-CB-Palavras-de-ensino.csv',
    'Probes-BC-CB-constant-words-3' : 'Ciclo3-4-Sondas-BC-CB-Palavras-reservadas.csv',
    'Probes-CD-generalization-and-constant-words-3' : 'Ciclo3-5-Sondas-CD-Palavras-generalizacao-reservadas.csv',
    'Probes-AC-generalization-and-constant-words-3' : 'Ciclo3-6-Sondas-AC-Palavras-generalizacao-reservadas.csv',

    'Probes-CD-12-training-8-generalization-words-4' : 'Ciclo4-7-Sondas-CD-Palavras-12-ensino-8-generalizacao.csv',
    'Training-AB-4' : 'Ciclo4-1-Treino-AB.csv',
    'Training-AC-4' : 'Ciclo4-2a-Treino-AC-CD.csv',
    'Training-AC-cumulative-4' : 'Ciclo4-2b-Treino-AC-Ref-Intermitente.csv',
    'Probes-BC-CB-training-words-4' : 'Ciclo4-3-Sondas-BC-CB-Palavras-de-ensino.csv',
    'Probes-BC-CB-constant-words-4' : 'Ciclo4-4-Sondas-BC-CB-Palavras-reservadas.csv',
    'Probes-CD-generalization-and-constant-words-4' : 'Ciclo4-5-Sondas-CD-Palavras-generalizacao-reservadas.csv',
    'Probes-AC-generalization-and-constant-words-4' : 'Ciclo4-6-Sondas-AC-Palavras-generalizacao-reservadas.csv',

    'Probes-CD-12-training-8-generalization-words-5' : 'Ciclo5-7-Sondas-CD-Palavras-12-ensino-8-generalizacao.csv',
    'Training-AB-5' : 'Ciclo5-1-Treino-AB.csv',
    'Training-AC-5' : 'Ciclo5-2a-Treino-AC-CD.csv',
    'Training-AC-cumulative-5' : 'Ciclo5-2b-Treino-AC-Ref-Intermitente.csv',
    'Probes-BC-CB-training-words-5' : 'Ciclo5-3-Sondas-BC-CB-Palavras-de-ensino.csv',
    'Probes-BC-CB-constant-words-5' : 'Ciclo5-4-Sondas-BC-CB-Palavras-reservadas.csv',
    'Probes-CD-generalization-and-constant-words-5' : 'Ciclo5-5-Sondas-CD-Palavras-generalizacao-reservadas.csv',
    'Probes-AC-generalization-and-constant-words-5' : 'Ciclo5-6-Sondas-AC-Palavras-generalizacao-reservadas.csv',

    'Probes-CD-12-training-8-generalization-words-6' : 'Ciclo6-7-Sondas-CD-Palavras-12-ensino-8-generalizacao.csv',
    'Training-AB-6' : 'Ciclo6-1-Treino-AB.csv',
    'Training-AC-6' : 'Ciclo6-2a-Treino-AC-CD.csv',
    'Training-AC-cumulative-6' : 'Ciclo6-2b-Treino-AC-Ref-Intermitente.csv',
    'Probes-BC-CB-training-words-6' : 'Ciclo6-3-Sondas-BC-CB-Palavras-de-ensino.csv',
    'Probes-BC-CB-constant-words-6' : 'Ciclo6-4-Sondas-BC-CB-Palavras-reservadas.csv',
    'Probes-CD-generalization-and-constant-words-6' : 'Ciclo6-5-Sondas-CD-Palavras-generalizacao-reservadas.csv',
    'Probes-AC-generalization-and-constant-words-6' : 'Ciclo6-6-Sondas-AC-Palavras-generalizacao-reservadas.csv'}