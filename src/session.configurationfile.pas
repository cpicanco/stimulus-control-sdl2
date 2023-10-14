{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

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
  , session.trials.shuffler
  ;

type
  TStartAt = record
    Trial : integer;
    Block  : integer;
  end;
  { TConfigurationFile }

  TConfigurationFile = class(TIniFile)
  private
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
    function GetBlock(BlockIndex : integer): TBlockData;
    function GetTrial(BlockIndex, TrialIndex : integer): TTrialData;
    function GetTrialBase(BlockIndex, TrialIndex : integer): TTrialData;
    procedure CopySection(AFrom, ATo : string; AConfigurationFile : TConfigurationFile);
    procedure WriteSection(ASectionName:string; ASection : TStrings);
    procedure AddNamesTo(AReferenceList : TReferenceList; ACurrentBlock : TBlockData);
  public
    constructor Create(const AConfigurationFile: string; AEscapeLineFeeds:Boolean=False); override;
    destructor Destroy; override;
    class function FullTrialSection(ABlock, ATrial : integer) : string;
    function ReadTrialString(ABlock : integer; ATrial : integer; AName:string):string;
    function ReadTrialInteger(ABlock : integer; ATrial : integer; AName:string):LongInt;
    function CurrentBlock: TBlockData;
    function CurrentTrial: TTrialData;
    function BeginTableName : string;
    function EndTableName : string;
    procedure Invalidate;
    procedure NewTrialOrder(ACurrentBlock : TBlockData);
    procedure ReadPositionsInBlock(ABlock:integer; APositionsList : TStrings);
    procedure WriteToBlock(ABlock : integer;AName, AValue: string);
    procedure WriteToTrial(ATrial : integer; AStrings : TStrings); overload;
    procedure WriteToTrial(ATrial : integer; AName, AValue: string); overload;
    procedure WriteToTrial(ATrial : integer; ABlock : integer; AName, AValue: string); overload;
    procedure WriteToInstruction(ATrial : integer; ABlock : integer; AName, AValue: string);

    procedure WriteToMain(AKey: string; AValue: string);
    procedure WriteMain(AMain : TStrings);
    procedure WriteBlockFromTarget(ATargetBlock : integer; ATargetConfigurationFile : TConfigurationFile;
      AlsoAppendTrials : Boolean = True);
    procedure WriteTrialFromTarget(ATargetBlock,ATargetTrial: integer; ATargetConfigurationFile : TConfigurationFile);
    procedure WriteBlockIfEmpty(ABlock : integer; ABlockSection : TStrings);
    //procedure WriteBlock(ABlock: TBlockData; AlsoAppendTrials: Boolean);
    //procedure WriteTrial(ATrial : TTrialData);
    property Blocks : integer read GetBlockCount;
    property TotalBlocks : integer read FBlockCount;
    property TotalTrials : integer read FTrialCount;
    property Trials[BlockIndex : integer] : integer read GetTrialCount;
    property Block[BlockIndex : integer] : TBlockData read GetBlock {write SetBlock};
    property Trial[BlockIndex, TrialIndex : integer] : TTrialData read GetTrial {write SetTrial};
    property StartAt : TStartAt read GetStartAt write SetStartAt;
  end;

var
  ConfigurationFile : TConfigurationFile;

implementation

uses StrUtils
  , session.constants
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

function TConfigurationFile.CurrentBlock: TBlockData;
begin
  Result := Block[Pool.Block.ID];
end;

function TConfigurationFile.CurrentTrial: TTrialData;
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

function TConfigurationFile.GetBlock(BlockIndex: integer): TBlockData;
var
  LBlockSection : string;
begin
  LBlockSection := BlockSection(BlockIndex);
  with Result, BlockKeys do
    begin
      ID := BlockIndex;
      TotalTrials:= Self.Trials[BlockIndex];
      Name:= ReadString(LBlockSection, _Name, '');

      NextBlockOnNotCriterion :=
        ReadInteger(LBlockSection, NextBlockOnNotCriterionKey, -1);
      BackUpBlockErrors :=
        ReadInteger(LBlockSection, BackUpBlockErrorsKey, 0);
      MaxBlockRepetition :=
        ReadInteger(LBlockSection, MaxBlockRepetitionKey, 0);
      MaxBlockRepetitionInSession :=
        ReadInteger(LBlockSection, MaxBlockRepetitionInSessionKey, 0);
      EndSessionOnHitCriterion :=
        ReadBool(LBlockSection, EndSessionOnHitCriterionKey, False);
      NextBlockOnHitCriterion :=
        ReadInteger(LBlockSection, NextBlockOnHitCriterionKey, -1);
      CrtHitPorcentage :=
        ReadInteger(LBlockSection, CrtHitPorcentageKey, 0);

      // old, not active
      ITI:= ReadInteger(LBlockSection, _ITI, 0);
      BkGnd:= ReadInteger(LBlockSection, _BkGnd, 0);
      DefNextBlock:=
        ReadString(LBlockSection, _DefNextBlock, '');
      MaxCorrection:=
        ReadInteger(LBlockSection, _MaxCorrection, 0);
      Counter:=
        ReadString(LBlockSection, _Counter, 'NONE');
      AutoEndSession :=
        ReadBool(LBlockSection, _AutoEndSession, False);
      CrtConsecutiveHit :=
        ReadInteger(LBlockSection, _CrtConsecutiveHit, 0);
      CrtConsecutiveMiss :=
        ReadInteger(LBlockSection, _CrtConsecutiveMiss, 0);
      CrtConsecutiveHitPerType :=
        ReadInteger(LBlockSection, _CrtConsecutiveHitPerType, 0);
      CrtHitValue :=
        ReadInteger(LBlockSection, _CrtHitValue, 0);
      CrtMaxTrials:=
        ReadInteger(LBlockSection, _CrtMaxTrials, 0);
      CrtCsqHit :=
        ReadInteger(LBlockSection, _CsqCriterion, 0);
    end;
end;

function TConfigurationFile.GetTrial(BlockIndex, TrialIndex: integer): TTrialData;
var
  LTrialSection : string;
  LInstructionSection : string;
  LParameters : TStringList;
  i : integer;
begin
  i := FPositions.Value(TrialIndex);
  if (i < 0) or
     (i >= TotalTrials) then begin
    raise EArgumentOutOfRangeException.Create(
      i.ToString + ' is out of bounds ' + TotalTrials.ToString);
  end;

  // do not shuffle instructions
  LInstructionSection := InstructionSection(BlockIndex, TrialIndex);

  // shuffle trials
  LTrialSection := TrialSection(BlockIndex, i);
  with Result do
    begin
      Id :=  i + 1;
      Kind := ReadString(LTrialSection, _Kind, '');
      ReferenceName := ReadString(LTrialSection, 'ReferenceName', '');
      Parameters := TStringList.Create;
      Parameters.CaseSensitive := False;
      Parameters.Duplicates := dupIgnore;
      ReadSectionValues(LTrialSection, Parameters);
      LParameters := TStringList.Create;
      try
        ReadSectionValues(LInstructionSection, LParameters);
        for i := 0 to LParameters.Count-1 do begin
          Parameters.Append(LParameters[i]);
        end;
      finally
        LParameters.Free;
      end;
    end;
end;

function TConfigurationFile.GetTrialBase(BlockIndex,
  TrialIndex: integer): TTrialData;
var
  LTrialSection : string;
begin
  LTrialSection := TrialSection(BlockIndex, TrialIndex);
  with Result do
    begin
      Id :=  TrialIndex + 1;
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

procedure TConfigurationFile.AddNamesTo(AReferenceList: TReferenceList;
  ACurrentBlock : TBlockData);
var
  i: Integer;
  LItem : TItem;
  LTrialData : TTrialData;
begin
  for i := 0 to ACurrentBlock.TotalTrials-1 do begin
    LTrialData := GetTrialBase(ACurrentBlock.ID, i);
    LItem.ReferenceName := LTrialData.ReferenceName;
    LItem.ID := i;
    AReferenceList.Add(LItem);
  end;
end;

procedure TConfigurationFile.Invalidate;
var
  i: Integer;
begin
  WriteInteger(_Main, _NumBlock, Blocks);
  for i := 0 to Blocks-1 do
    WriteString(BlockSection(i),_NumTrials, Trials[i].ToString+' 1');
end;

procedure TConfigurationFile.NewTrialOrder(ACurrentBlock : TBlockData);
var
  FReferenceList : TReferenceList;
begin
  FReferenceList := TReferenceList.Create;
  try
    AddNamesTo(FReferenceList, ACurrentBlock);
    FPositions.Shuffle(FReferenceList);
  finally
    FReferenceList.Free;
  end;
end;

procedure TConfigurationFile.ReadPositionsInBlock(ABlock: integer;
  APositionsList: TStrings);
var
  L : TStringList;
  LNumComp: LongInt;
  LTrialSection, LKeyName, S: String;
  j, i: Integer;
begin
  L := TStringList.Create;
  L.Sorted := True;
  L.Duplicates:=dupIgnore;
  try
    for i := 0 to Trials[ABlock]-1 do
      begin
        LTrialSection := TrialSection(ABlock, i);

        // sample
        if ReadString(LTrialSection,_Kind,'') = T_MTS then
          begin
            LKeyName := _Samp+_cBnd;
            S := ReadString(LTrialSection,LKeyName,'');
            if S <> '' then
              L.Append(S);
          end;

        // comparisons
        LNumComp := ReadInteger(LTrialSection,_NumComp,0);
        if LNumComp > 0 then
          for j := 0 to  LNumComp-1 do
            begin
              LKeyName := _Comp+IntToStr(j+1)+_cBnd;
              S := ReadString(LTrialSection,LKeyName,'');
              if S <> '' then
                L.Append(S);
            end;
      end;

    j := 0;
    for i := L.Count-1 downto 0 do
      begin
        APositionsList.Values[IntToStr(j+1)] := L[i];
        Inc(j);
      end;

  finally
    L.Free;
  end;
end;

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
  FBlockCount := 0;
  GetBlockCount;
  FPositions := TShuffler.Create;
end;

destructor TConfigurationFile.Destroy;
begin
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

procedure TConfigurationFile.WriteToInstruction(ATrial: integer;
  ABlock: integer; AName, AValue: string);
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

