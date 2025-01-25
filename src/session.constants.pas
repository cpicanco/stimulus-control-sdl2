{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants;

{$mode objfpc}{$H+}

interface

const
  DefaultPupilAddress = '127.0.1.1:50020';

const
  clDarkGreen = $006400;

const
  // Posições
  _Positions = 'Positions';
  _Cols = 'Cols';
  _Rows = 'Rows';
  _NumPos = 'NumPos';

  // Main
  _Main = 'Main';
  _Name = 'Name';
  _Subject = 'Subject';
  _Media  = 'Media';
  _Data  = 'Data';
  _RootData   = 'RootData';
  _RootMedia  = 'RootMedia';
  _NumBlock = 'NumBlock';
  _Type = 'Type';
  _ServerAddress = 'ServerAddress';

  // Block
  _NumTrials = 'NumTrials';
  _AutoEndSession = 'AutoEndSession';
  _CrtHitPorcentage = 'Criteria.HitPorcentage';
  _CrtHitValue = 'Criteria.HitValue';
  _CrtConsecutiveHit  = 'ConsecutiveHitCriterion';
  _CrtConsecutiveHitPerType = 'ConsecutiveHitCriterionPerType';
  _CrtConsecutiveMiss = 'ConsecutiveMissCriterion';
  _CrtMaxTrials = 'MaxTrialCriterion';
  _NextBlockOnCriteria = 'NextBlock.OnCriteria';
  _NextBlockOnNotCriteria = 'NextBlock.OnNotCriteria';
  _CsqCriterion = 'CsqCriterion';
  _DefNextBlock = 'DefNextBlock';
  _MaxCorrection = 'MaxCorrection';
  _MaxBlockRepetition = 'MaxBlockRepetition';
  _ITI = 'ITI';
  _Counter = 'Counter';
  _Counter_Performance = 'performance';

  // Block and Trial
  _BkGnd = 'BkGnd';

  // Trial
  _Cursor = 'Cursor';
  _AutoNxt = 'AutoNxt';
  _CustomNxtValue = 'CustomNxtValue';
  _Kind = 'Kind';
  _LimitedHold = 'LimitedHold';
  _Consequence = 'Consequence';
  _ShowCounter = 'Counter';
  _CounterType = 'CounterType';

  // MRD
  _UseMedia = 'UseMedia';
  _ShowStarter = 'Starter';
  _Angle = 'Angle';
  _ExpectedResponse = 'Response';
  _Schedule = 'Schedule';
  _NextTrial = 'NextTrial';
  _cGap = 'Gap';                 //boolean
  _cGap_Degree = 'GapDegree';
  _cGap_Length = 'GapLength';
  //some constants are shared with 'SIMPLE' when 'usemedia' is true

  // MSG
  _Msg = 'Msg';
  _MsgWidth = 'MsgWidth';
  _MsgFontSize = 'MsgFontSize';
  _MsgFontColor = 'MsgFontColor';
  _Prompt = 'Prompt';
  _Style = 'Style';

  // MTS
  _SampleType = 'Delayed';
  _Delay = 'Delay';

  // TCLB
  _ShowDots = 'ShowDots';
  _Blocking = 'Blocking';

  // FPE
  _DrawingType = 'DrawingType';
  _Contingency = 'Contingency';
  _ShouldPlaySound = 'ShouldPlaySound';

  // GNG
  _PopUpTime = 'ConsequenceTime';
  _PresentConsequenceJustOnce = 'PresentConsequenceJustOnce';
  _ScreenSide = 'ScreenSide';
  _ResponseStyle = 'ResponseStyle';
  _Category = 'Category';

  // SIMPLE
  _NumComp = 'NumComp';   // number of stimuli on screen / comparisons
  _cBnd = 'Bnd';          // top left width height
  _cCap = 'Cap';
  _cStm = 'Stm';          // stimulus path
  _cSch = 'Sch';          // reinforcement schedule
  _cMsg = 'Msg';          // message sent to the report {REGDATA}
  _cCsq = 'Csq';          // PLP code,see the unit 'library'
  _cUsb = 'Usb';          // RS232 code, see the unit 'library'
  _cRes = 'Res';          // Kind of the response, hit, miss or none
  _cNxt = 'Nxt';          // Next trial {if = '0', same as '1', if > 0, go to the specified trial}
  _cIET = 'IET';          // Inter Trial Interval, in miliseconds
  _cTO  = 'TO';           // Time-Out Interval, in miliseconds

  // LIKERT
  _Left = 'Left';
  _Right = 'Right';

  // BINARY CHOICE
  _Now     = 'Now';
  _LastNow = 'LastNow';

  // Identificadores
  _Kplus = 'K+';
  _Kminus = 'K-';
  _Pos = 'P';
  _Block = 'Block';
  _Trial = 'T';
  _Comp = 'C';
  _Samp = 'S';

  LAST_BLOCkS_INI_FILENAME = 'last.blocks.ini';
  LAST_BLOCk_INI_FILENAME = 'last.block.ini';


const
  { escriba }
  KEnter = #13#10;
  KPositions = '[Positions]' + #13#10;
  KMain = '[Main]' + #13#10;
  KName = 'Name=' + #9;
  KSubject = 'Subject=' + #9;
  KType = 'Type=' + #9;
  KMedia = 'Media=' + #9;
  KRootMedia = 'RootMedia=' + #9;
  KData = 'Data=' + #9;
  KRootData = 'RootData=' + #9;
  KServer = 'ServerAddress=' + #9;
  KNumBlock = 'NumBlock=' + #9;
  KNumTrial = 'NumTrials=' + #9;
  KMaxCorrection = 'MaxCorrection=' + #9;
  KCrtConsecutiveHit = 'ConsecutiveHitCriterion=' + #9;
  KCrtConsecutiveMiss = 'ConsecutiveMissCriterion=' + #9;
  KCrtMaxTrials = 'MaxTrialCriterion=' + #9;
  KCrtKCsqHit = 'CsqCriterion=' + #9;
  KNumComp = 'NumComp=' + #9;
  KRows = 'Rows=' + #9;
  KCols = 'Cols=' + #9;
  KNumPos = 'NumPos=' + #9;
  KBackGround = 'BkGnd=' + #9;
  KITInterval = 'ITI=' + #9;
  KCursor = 'Cursor=' + #9;
  KKind = 'Kind=' + #9;
  KComAtraso = 'Delayed=' + #9;
  KAtraso = 'Delay=' + #9;
  KAutoNext  = 'AutoNxt=' + #9;
  KCustomNext = 'CustomNxtValue=' + #9;
  KWidth = 'MsgWidth=' + #9;
  KFontSize = 'MsgFontSize=' + #9;
  KFontColor = 'MsgFontColor=' + #9;
  KPrompt = 'Prompt=' + #9;

  KUseMedia = 'UseMedia=' + #9;  //boolean
  KContingency = 'Contingency=' + #9;
  KShowStarter = 'Starter=' + #9; //boolean
  KAngle = 'Angle=' + #9;
  KExpectedResponse = 'Response=' + #9;
  KLimitedHold = 'LimitedHold=' + #9;
  KSchedule = 'Schedule=' + #9;
  KNextTrial = 'NextTrial=' + #9;
  KcGap = 'Gap=' + #9;                 //boolean
  KcGap_Degree = 'GapDegree=' + #9;
  KcGap_Length = 'GapLength=' + #9;
  KConsequence = 'Consequence=' +#9;

  KBnd = 'Bnd=' + #9;
  KStm = 'Stm=' + #9;
  KIET = 'IET=' + #9;
  KSch = 'Sch=' + #9;
  KMsg = 'Msg=' + #9;
  KCsq = 'Csq=' + #9;
  KUsb = 'Usb=' + #9;
  KRes = 'Res=' + #9;
  KNxt = 'Nxt=' + #9;
  KTO = 'TO='  + #9;

  { Session Parameters }
  T_CIC = 'CIC';
  T_CRT = 'CRT';
  T_CND = 'CND';

  { General Parameters }
  T_END = 'END';

  { Trial Kinds }

  T_FOC = 'FOC';
  T_GNG = 'GNG';
  T_MSQ = 'MSQ';
  T_DZT = 'DZT';
  T_CLB = 'CLB';
  T_FPFN = 'FPFN';
  T_MRD = 'MRD';
  T_MSG = 'MSG';
  T_HTMLMESSAGE = 'HTMLMESSAGE';
  T_RTF = 'RTFM';
  T_EO1 = 'OP1';
  T_VTFI = 'VTFI';
  T_TMB = 'TMB';
  T_CHO = 'CHOICE';
  T_INP = 'TXTINP';
  T_MTS = 'MTS';
  T_DMTS = 'DMTS';
  T_DMTSA = 'DMTSA';
  T_DRAGDROP = 'DRAGDROP';
  T_Simple = 'SIMPLE';
  T_LIK = 'LIKERT';
  T_PFR = 'PERFORMANCE';
  T_BAT = 'BEFORE_AFTER';
  T_CTX = 'EQUAL_DIFFERENT';
  T_RFT = 'BEFORE_AFTER_EQUAL_DIFFERENT';

  { Schedules }
  T_RR  = 'RR';
  T_FR  = 'FR';
  T_VR = 'VR';
  T_RI = 'RI';
  T_FI = 'FI';
  T_VI  = 'VI';
  T_RT = 'RT';
  T_VT = 'VT';
  T_FT  = 'FT';
  T_CRF = 'CRF';
  T_EXT  = 'EXT';
  T_DRL = 'DRL';
  T_DRH = 'DRH';

  { Res }
  T_HIT = 'HIT';
  T_MISS = 'MISS';
  T_NONE = 'NONE';

  { Nxt }
  T_REP = 'CRT';

const
  //report  messages
  rmKeyPlus = 'Tecla +';
  rmKeyMinus = 'Tecla -';

// report headers
const
  HeaderTabs : string = #9;

resourcestring
  ExceptionNoScheduleFound = 'Nenhum esquema de reforço encontrado, revise o arquivo de configuração.';
  ExceptionConfigurationNotAllowed = 'A configuração não é permitida.';

  rsReportDelayBeg = 'Atraso.Inicio';
  rsReportDelayEnd = 'Atraso.Fim';

  rsReportPoints = 'Pontos';
  rsReportMsgTxt = 'Mensagem';
  rsReportStmBeg = 'S.Inicio';
  rsReportStmDur = 'S.Duracao';
  rsReportStmEnd = 'S.Fim';
  rsReportStmEndR = 'S.Fim.Reforco';
  rsReportStmEndO = 'S.Fim.Omissao';
  rsReportStmLeft = 'S.Comparacao.Esquerda';
  rsReportStmRight = 'S.Comparacao.Direita';

  rsReportRspLat = 'R.Latencia';
  rsReportRspFrq = 'R.Frequencia';
  rsReportRspExp = 'R.Prevista';

  rsReportRspStl = 'R.Esperada';
  rsReportScrSid = 'S.PosicaoNaTela';

  rsReportStmCmp = 'S.Comparacoes';
  rsReportStmCmpBeg = 'S.Comparacoes.Inicio';
  rsReportStmCmpEnd = 'S.Comparacoes.Fim';
  rsReportRspCmp = 'R.Comparacao.Escolhida';
  rsReportRspCmpAlt = 'R.Comparacao.NaoEscolhida';
  rsReportRspCmpLat = 'R.Comparacao.Escolhida.Latencia';
  rsReportStmCmpDur = 'S.Comparacoes.Duracao';

  rsReportStmMod = 'S.Modelo';
  rsReportStmModBeg = 'S.Modelo.Inicio';
  rsReportStmModEnd = 'S.Modelo.Fim';
  rsReportRspModLat = 'R.Modelo.Latencia';
  rsReportStmModDur = 'S.Modelo.Duracao';
  rsReportStmModDel = 'S.Modelo.Atraso';
  rsReportRspModFrq = 'R.Modelo.Frequencia';

  rsReportCsqPLP = 'C.PLP';
  rsReportCsqUSB = 'C.USB';
  rsReportCsqRes = 'C.Resultado';
  rsReportRspCmpFrq = 'R.Comparacoes.Frequencia';

  rsReportTime = 'Tempo';
  rsReportBlockID  = 'Bloco.ID';
  rsReportBlockName = 'Bloco.Nome';
  rsReportTrialID = 'Tentativa.ID';
  rsReportTrialNO = 'Tentativa.Contador';
  rsReportTrialName = 'Tentativa.Nome';
  rsReportEvent = 'Evento';
  rsReportITIBeg = 'IET.Inicio';
  rsReportITIEnd = 'IET.Fim';
  rsReportITI = 'IET';

  rsReportRspLft = 'Lft.Cmp.';
  rsReportRspTop = 'Top.Cmp.';

  rsNA = 'NA';

implementation


end.

