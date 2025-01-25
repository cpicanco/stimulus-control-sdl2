{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.configurationfile;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils
  , Session.Configuration
  , IniFiles
  , session.shuffler.base
  ;

type

  TStartAt = record
    Trial : integer;
    Block  : integer;
  end;

  { TConfigurationFile }

  TConfigurationFile = class(TIniFile)
  private
    FCurrentTrialParameters : TStringList;
    FPositions : TShuffler;
    FBlockCount : integer;
    FTrialCount: integer;
    function GetStartAt: TStartAt;
    procedure SetStartAt(AValue: TStartAt);
    class function InstructionSection(BlockIndex, TrialIndex : integer) : string;
    class function TrialSection(BlockIndex, TrialIndex : integer) : string;
    class function BlockSection(BlockIndex : integer) : string;
    function CurrentBlockSection : string;
    function GetBlockCount : integer;
    function GetTrialCount(BlockIndex : integer): integer;
    function GetBlock(BlockIndex : integer): TBlockConfiguration;
    function GetTrial(BlockIndex, TrialIndex : integer): TTrialConfiguration;
    function GetTrialBase(BlockIndex, TrialIndex : integer): TTrialConfiguration;
    procedure CopySection(AFrom, ATo : string; AConfigurationFile : TConfigurationFile);
    procedure WriteSection(ASectionName:string; ASection : TStrings);
  public
    constructor Create(const AConfigurationFile: string; AEscapeLineFeeds:Boolean=False); override;
    destructor Destroy; override;
    class function FullTrialSection(ABlock, ATrial : integer) : string;
    function ReadTrialString(ABlock : integer; ATrial : integer; AName:string):string;
    function ReadTrialInteger(ABlock : integer; ATrial : integer; AName:string):LongInt;
    function CurrentBlock: TBlockConfiguration;
    function CurrentTrial: TTrialConfiguration;
    function BeginTableName : string;
    function EndTableName : string;
    procedure Invalidate;
    procedure AppendSectionValues(ASection : string; AParameters : TStringList);
    procedure NewOrdereringForTrialsInBlock(
      ACurrentBlock : TBlockConfiguration);
    //procedure ReadPositionsInBlock(ABlock:integer; APositionsList : TStrings);
    procedure WriteToBlock(ABlock : integer;AName, AValue: string);
    procedure WriteToTrial(ATrial : integer; AStrings : TStrings); overload;
    procedure WriteToTrial(ATrial : integer; AName, AValue: string); overload;
    procedure WriteToTrial(ATrial : integer; ABlock : integer; AName, AValue: string); overload;
    procedure WriteToInstruction(ABlock : integer; ATrial : integer; AName, AValue: string);

    procedure WriteToMain(AKey: string; AValue: string);
    procedure WriteMain(AMain : TStrings);
    procedure WriteBlockFromTarget(ATargetBlock : integer; ATargetConfigurationFile : TConfigurationFile;
      AlsoAppendTrials : Boolean = True);
    procedure WriteTrialFromTarget(ATargetBlock,ATargetTrial: integer; ATargetConfigurationFile : TConfigurationFile);
    procedure WriteBlockIfEmpty(ABlock : integer; ABlockSection : TStrings);
    //procedure WriteBlock(ABlock: TBlockConfiguration; AlsoAppendTrials: Boolean);
    //procedure WriteTrial(ATrial : TTrialConfiguration);
    property Blocks : integer read GetBlockCount;
    property TotalBlocks : integer read FBlockCount;
    property TotalTrials : integer read FTrialCount;
    property Trials[BlockIndex : integer] : integer read GetTrialCount;
    property Block[BlockIndex : integer] : TBlockConfiguration read GetBlock {write SetBlock};
    property Trial[BlockIndex, TrialIndex : integer] : TTrialConfiguration read GetTrial {write SetTrial};
    property StartAt : TStartAt read GetStartAt write SetStartAt;
  end;

var
  ConfigurationFile : TConfigurationFile;

implementation

uses StrUtils
  , session.shuffler.types
  , session.dynamics.base
  , session.constants
  , session.constants.trials
  , session.constants.blocks
  , session.pool;

{ TConfigurationFile }

function TConfigurationFile.GetBlockCount: integer;
begin
  FBlockCount := 0;
  while SectionExists(BlockSection(FBlockCount)) do
    Inc(FBlockCount);
  Result := FBlockCount;
end;

class function TConfigurationFile.BlockSection(BlockIndex: integer): string;
begin
  Result := _Block + #32 + IntToStr(BlockIndex+1);
end;

function TConfigurationFile.GetTrialCount(BlockIndex : integer): integer;
begin
  FTrialCount := 0;
  while SectionExists(TrialSection(BlockIndex, FTrialCount)) do
    Inc(FTrialCount);
  Result := FTrialCount;
end;

class function TConfigurationFile.TrialSection(BlockIndex,
  TrialIndex: integer): string;
begin
  Result := BlockSection(BlockIndex) + ' - ' + _Trial + IntToStr(TrialIndex+1);
end;

function TConfigurationFile.GetStartAt: TStartAt;
var
  S : string;
begin
  S := ReadString(_Main, 'StartAt', '1-1');
  Result.Block := ExtractDelimited(1, S, ['-']).ToInteger-1;
  Result.Trial := ExtractDelimited(2, S, ['-']).ToInteger-1;
end;

procedure TConfigurationFile.SetStartAt(AValue: TStartAt);
begin
  WriteToMain('StartAt',
    (AValue.Block+1).ToString + '-' + (AValue.Trial+1).ToString);
end;

class function TConfigurationFile.InstructionSection(BlockIndex,
  TrialIndex: integer): string;
begin
  Result := BlockSection(BlockIndex) + ' - ' + 'M' + IntToStr(TrialIndex+1);
end;

function TConfigurationFile.CurrentBlock: TBlockConfiguration;
begin
  Result := Block[Pool.Block.ID];
end;

function TConfigurationFile.CurrentTrial: TTrialConfiguration;
begin
  Result := Trial[
    Pool.Block.ID,
    Pool.Trial.ID];
end;

function TConfigurationFile.BeginTableName: string;
begin
  Result := ReadString(CurrentBlockSection, 'BeginTable', '');
end;

function TConfigurationFile.EndTableName: string;
begin
  Result := ReadString(CurrentBlockSection, 'EndTable', '');
end;

function TConfigurationFile.CurrentBlockSection: string;
begin
  Result := BlockSection(Pool.Block.ID);
end;

function TConfigurationFile.GetBlock(BlockIndex: integer): TBlockConfiguration;
var
  Ltest1 : string;
  Ltest2 : string;
  LBlockSection : string;

  LRepeatStyle : TBlockRepeatStyle =
    TBlockRepeatStyle.None;

  LEndCriterionStyle : TBlockEndCriterionStyle =
    TBlockEndCriterionStyle.HitPorcentage;

  LBlockEndCriterionEvaluationTime : TBlockEndCriterionEvaluationTime =
    TBlockEndCriterionEvaluationTime.OnBlockEnd;
begin
  LBlockSection := BlockSection(BlockIndex);
  with Result, ParserBlockKeys do begin
    ID := BlockIndex;
    TotalTrials:= Self.Trials[BlockIndex];
    Name:= ReadString(LBlockSection, NameKey, '');

    EndSessionOnCriterion := ReadBool(
      LBlockSection, EndSessionOnCriterionKey, False);

    EndSessionOnNotCriterionAfterBlockRepetitions := ReadBool(
      LBlockSection, EndSessionOnNotCriterionAfterBlockRepetitionsKey, False);

    RepeatStyle := ReadString(
      LBlockSection, RepeatStyleKey,
      LRepeatStyle.ToString).ToRepeatStyle;

    EndCriterionStyle := ReadString(
      LBlockSection, EndCriterionStyleKey,
      LEndCriterionStyle.ToString).ToEndCriterionStyle;

    EndCriterionEvaluationTime := ReadString(
      LBlockSection, EndCriterionEvaluationTimeKey,
      LBlockEndCriterionEvaluationTime.ToString).ToEndCriterionEvaluationTime;

    MaxBlockRepetitionConsecutives := ReadInteger(
      LBlockSection, MaxBlockRepetitionConsecutivesKey, 0);

    MaxBlockRepetitionInSession := ReadInteger(
      LBlockSection, MaxBlockRepetitionInSessionKey, 0);

    NextBlockOnCriterion := ReadInteger(
      LBlockSection, NextBlockOnCriterionKey, -1);

    NextBlockOnNotCriterion := ReadInteger(
      LBlockSection, NextBlockOnNotCriterionKey, -1);

    EndCriterionValue := ReadInteger(
      LBlockSection, EndCriterionValueKey, 0);

    Reinforcement := ReadInteger(
      LBlockSection, ReinforcementKey, 100);

    // old, not active
    //ITI:= ReadInteger(LBlockSection, _ITI, 0);
    //BkGnd:= ReadInteger(LBlockSection, _BkGnd, 0);
    //DefNextBlock:=
    //  ReadString(LBlockSection, _DefNextBlock, '');
    //MaxCorrection:=
    //  ReadInteger(LBlockSection, _MaxCorrection, 0);
    //Counter:=
    //  ReadString(LBlockSection, _Counter, 'NONE');
    //AutoEndSession :=
    //  ReadBool(LBlockSection, _AutoEndSession, False);
    //CrtConsecutiveMiss :=
    //  ReadInteger(LBlockSection, _CrtConsecutiveMiss, 0);
    //CrtConsecutiveHitPerType :=
    //  ReadInteger(LBlockSection, _CrtConsecutiveHitPerType, 0);
    //CrtHitValue :=
    //  ReadInteger(LBlockSection, _CrtHitValue, 0);
    //CrtMaxTrials:=
    //  ReadInteger(LBlockSection, _CrtMaxTrials, 0);
    //CrtCsqHit :=
    //  ReadInteger(LBlockSection, _CsqCriterion, 0);
  end;
end;

function TConfigurationFile.GetTrial(BlockIndex, TrialIndex: integer): TTrialConfiguration;
var
  LTrialSection : string;
  LInstructionSection : string;
  i : integer;
begin
  i := FPositions.Values(TrialIndex);
  if (i < 0) or
     (i >= TotalTrials) then begin
    raise EArgumentOutOfRangeException.Create(
      i.ToString + ' is out of bounds ' + TotalTrials.ToString);
  end;

  // do not shuffle instructions
  LInstructionSection := InstructionSection(BlockIndex, TrialIndex);

  // shuffle trials
  LTrialSection := TrialSection(BlockIndex, i);
  FCurrentTrialParameters.Clear;
  with Result, TrialKeys do begin
    Id :=  i;
    Kind := ReadString(LTrialSection, KindKey, '');
    ReferenceName := ReadString(LTrialSection, ReferenceNameKey, '');
    ReadSectionValues(LTrialSection, FCurrentTrialParameters);
    AppendSectionValues(LInstructionSection, FCurrentTrialParameters);
    SetTrialDynamics(FCurrentTrialParameters);
    Parameters := FCurrentTrialParameters;
  end;
end;

function TConfigurationFile.GetTrialBase(BlockIndex,
  TrialIndex: integer): TTrialConfiguration;
var
  LTrialSection : string;
begin
  LTrialSection := TrialSection(BlockIndex, TrialIndex);
  with Result do begin
    Id :=  TrialIndex;
    Kind := ReadString(LTrialSection, _Kind, '');
    ReferenceName := ReadString(LTrialSection, 'ReferenceName', '');
    Parameters := TStringList.Create;
    Parameters.CaseSensitive := False;
    Parameters.Duplicates := dupIgnore;
    ReadSectionValues(LTrialSection, Parameters);
  end;
end;

procedure TConfigurationFile.CopySection(AFrom, ATo: string;
  AConfigurationFile: TConfigurationFile);
var
  LSection : TStringList;
  LTargetSectionName,
  LSelfSectionName : string;
begin
  if AConfigurationFile.SectionExists(AFrom) then
    begin
      LSection := TStringList.Create;
      LSection.CaseSensitive := False;
      LSection.Duplicates := dupIgnore;
      try
        LTargetSectionName:= AFrom;
        LSelfSectionName := ATo;
        AConfigurationFile.ReadSectionValues(LTargetSectionName,LSection);
        WriteSection(LSelfSectionName,LSection);
      finally
        LSection.Clear;
        LSection.Free;
      end;
    end;
end;

procedure TConfigurationFile.WriteSection(ASectionName: string;
  ASection: TStrings);
var
  LLine, LKeyName: String;
begin
  for LLine in ASection do
    begin
      LKeyName := ASection.ExtractName(LLine);
      WriteString(ASectionName, LKeyName, ASection.Values[LKeyName]);
    end;
end;

procedure TConfigurationFile.Invalidate;
var
  i: Integer;
  s: string;
begin
  WriteInteger(_Main, _NumBlock, Blocks);
  for i := 0 to Blocks-1 do begin
    s:= Trials[i].ToString;
    WriteString(BlockSection(i),_NumTrials, s);
  end;
end;

procedure TConfigurationFile.AppendSectionValues(ASection: string;
  AParameters: TStringList);
var
  LParameters : TStringList;
  i: Integer;
begin
  LParameters := TStringList.Create;
  try
    ReadSectionValues(ASection, LParameters);
    for i := 0 to LParameters.Count-1 do begin
      AParameters.Append(LParameters[i]);
    end;
  finally
    LParameters.Free;
  end;
end;

procedure TConfigurationFile.NewOrdereringForTrialsInBlock(
  ACurrentBlock: TBlockConfiguration);
var
  LReferenceList : TReferenceList;
  procedure GetTrialsReferenceNames(AReferenceList: TReferenceList;
    ACurrentBlock : TBlockConfiguration);
  var
    i: Integer;
    LItem : TItem;
    LTrialData : TTrialConfiguration;
  begin
    for i := 0 to ACurrentBlock.TotalTrials-1 do begin
      LTrialData := GetTrialBase(ACurrentBlock.ID, i);
      try
        LItem.ReferenceName := LTrialData.ReferenceName;
        LItem.ID := i;
        AReferenceList.Add(LItem);
      finally
        LTrialData.Parameters.Clear;
        LTrialData.Parameters.Free;
      end;
    end;
  end;
begin
  LReferenceList := TReferenceList.Create;
  try
    GetTrialsReferenceNames(LReferenceList, ACurrentBlock);
    FPositions.Shuffle(LReferenceList);
  finally
    LReferenceList.Free;
  end;
end;

//procedure TConfigurationFile.ReadPositionsInBlock(ABlock: integer;
//  APositionsList: TStrings);
//var
//  L : TStringList;
//  LNumComp: LongInt;
//  LTrialSection, LKeyName, S: String;
//  j, i: Integer;
//begin
//  L := TStringList.Create;
//  L.Sorted := True;
//  L.Duplicates:=dupIgnore;
//  try
//    for i := 0 to Trials[ABlock]-1 do
//      begin
//        LTrialSection := TrialSection(ABlock, i);
//
//        // sample
//        if ReadString(LTrialSection,_Kind,'') = T_MTS then
//          begin
//            LKeyName := _Samp+_cBnd;
//            S := ReadString(LTrialSection,LKeyName,'');
//            if S <> '' then
//              L.Append(S);
//          end;
//
//        // comparisons
//        LNumComp := ReadInteger(LTrialSection,_NumComp,0);
//        if LNumComp > 0 then
//          for j := 0 to  LNumComp-1 do
//            begin
//              LKeyName := _Comp+IntToStr(j+1)+_cBnd;
//              S := ReadString(LTrialSection,LKeyName,'');
//              if S <> '' then
//                L.Append(S);
//            end;
//      end;
//
//    j := 0;
//    for i := L.Count-1 downto 0 do
//      begin
//        APositionsList.Values[IntToStr(j+1)] := L[i];
//        Inc(j);
//      end;
//
//  finally
//    L.Clear;
//    L.Free;
//  end;
//end;

function TConfigurationFile.ReadTrialString(ABlock: integer; ATrial: integer;
  AName: string): string;
begin
  Result := ReadString(TrialSection(ABlock, ATrial), AName, '');
end;

function TConfigurationFile.ReadTrialInteger(ABlock: integer; ATrial: integer;
  AName: string): LongInt;
begin
  Result := ReadInteger(TrialSection(ABlock, ATrial), AName, 0);
end;

constructor TConfigurationFile.Create(const AConfigurationFile: string;
  AEscapeLineFeeds: Boolean);
begin
  inherited Create(AConfigurationFile, AEscapeLineFeeds);
  BoolTrueStrings := TrueBoolStrs;
  BoolFalseStrings := FalseBoolStrs;
  FBlockCount := 0;
  GetBlockCount;
  FPositions := TShuffler.Create;
  FCurrentTrialParameters := TStringList.Create;
  FCurrentTrialParameters.CaseSensitive := False;
  FCurrentTrialParameters.Duplicates := dupIgnore;
end;

destructor TConfigurationFile.Destroy;
begin
  FCurrentTrialParameters.Free;
  FPositions.Free;
  inherited Destroy;
end;

class function TConfigurationFile.FullTrialSection(ABlock,
  ATrial: integer): string;
begin
  Result := '[' + TrialSection(ABlock, ATrial) + ']';
end;

procedure TConfigurationFile.WriteToBlock(ABlock: integer; AName, AValue: string);
begin
  WriteString(BlockSection(ABlock),AName,AValue);
end;

procedure TConfigurationFile.WriteToTrial(ATrial: integer; AStrings: TStrings);
begin
  WriteSection(TrialSection(Blocks,ATrial),AStrings);
end;

procedure TConfigurationFile.WriteToTrial(ATrial: integer;
  AName, AValue: string);
begin
  WriteString(TrialSection(Blocks,ATrial),AName,AValue);
end;

procedure TConfigurationFile.WriteToTrial(ATrial: integer; ABlock: integer;
  AName, AValue: string);
begin
  WriteString(TrialSection(ABlock, ATrial),AName,AValue);
end;

procedure TConfigurationFile.WriteToInstruction(ABlock: integer;
  ATrial: integer;AName, AValue: string);
begin
  WriteString(InstructionSection(ABlock, ATrial),AName,AValue);
end;

procedure TConfigurationFile.WriteToMain(AKey: string; AValue: string);
begin
  WriteString(_Main, AKey, AValue);
end;

procedure TConfigurationFile.WriteMain(AMain: TStrings);
begin
  WriteSection(_Main, AMain);
end;

procedure TConfigurationFile.WriteBlockFromTarget(ATargetBlock: integer;
  ATargetConfigurationFile: TConfigurationFile; AlsoAppendTrials: Boolean);
var
  LSelfSectionName,
  LTargetSectionName : string;
  i: integer;
begin
  LSelfSectionName := BlockSection(Blocks);
  LTargetSectionName := BlockSection(ATargetBlock);
  CopySection(LTargetSectionName,LSelfSectionName, ATargetConfigurationFile);
  if AlsoAppendTrials then
    if ATargetConfigurationFile.Trials[ATargetBlock] > 0 then
      for i := 0 to ATargetConfigurationFile.Trials[ATargetBlock]-1 do
        WriteTrialFromTarget(ATargetBlock,i+1,ATargetConfigurationFile);
end;

procedure TConfigurationFile.WriteTrialFromTarget(ATargetBlock,
  ATargetTrial: integer; ATargetConfigurationFile: TConfigurationFile);
var
  LSelfSectionName,
  LTargetSectionName : string;
begin
  LSelfSectionName := TrialSection(Blocks, Trials[Blocks]);
  LTargetSectionName:= TrialSection(ATargetBlock,ATargetTrial);
  CopySection(LTargetSectionName,LSelfSectionName, ATargetConfigurationFile);
end;

procedure TConfigurationFile.WriteBlockIfEmpty(ABlock: integer;
  ABlockSection: TStrings);
var
  LBlockSection,
  LLine, LKeyName: String;
  function EmptyKey : Boolean;
  var
    S : string;
  begin
    S := ReadString(LBlockSection, LKeyName, '');
    case Length(S) of
      0: Result := True;
      1: Result := not (S[1] in [#0..#32]);
      2..MaxInt : Result := False;
    end;
  end;

begin
  LBlockSection:=BlockSection(ABlock);
  for LLine in ABlockSection do
    begin
      LKeyName := ABlockSection.ExtractName(LLine);
      if ValueExists(LBlockSection, LKeyName) then
        begin
          if EmptyKey then
            WriteString(LBlockSection, LKeyName, ABlockSection.Values[LKeyName])
          else; // do nothing
        end
      else
        WriteString(LBlockSection, LKeyName, ABlockSection.Values[LKeyName]);
    end;
end;

finalization
  if Assigned(ConfigurationFile) then
    ConfigurationFile.Free;

end.

