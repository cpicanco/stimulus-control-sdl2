{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit experiments.trials;

{$mode objfpc}{$H+}

interface

uses session.constants.trials, session.constants.mts;

procedure WriteToConfigurationFile(AFilename : string);

var
  GlobalTrialParameters : TTrialParameters;
  MTSParameters   : TMTSParameters;

implementation

uses
  Classes
  , SysUtils
  , LazFileUtils
  //, StrUtils
  , session.csv
  , session.constants.blocks
  , sdl.app.output
  , session.constants.dragdrop
  //, session.fileutils
  , session.configurationfile
  , session.configurationfile.writer
  , sdl.app.trials.mts
  , picanco.experiments.words.types
  , picanco.experiments.words
  , picanco.experiments.images
  , picanco.experiments.audio
  , picanco.experiments.constants
  //, sdl.app.trials.dragdrop
  ;

var Writer : TConfigurationWriter;

procedure WriteTrials(AName: string; AReferenceName : string;
  ARelation: string; AWord: TWord; AComparisons: integer;
  AHasConsequence : Boolean = True; ASamples: integer = 1;
  ARepeatTrials: integer = 1; AHasLimitedHold : Boolean = False);
var
  LWord : TWord;
  i: Integer;
begin
  with Writer.TrialConfig do begin
    with TrialKeys do begin
      Values[ReferenceName] := AReferenceName;
      Values[Name] := AName;
      Values[Cursor] := GlobalTrialParameters.Cursor.ToString;
      Values[Kind] := TMTS.ClassName;
      if AHasLimitedHold then
        Values[LimitedHold] := GlobalTrialParameters.LimitedHold.ToString;
      Values[InterTrialInterval] := GlobalTrialParameters.InterTrialInterval.ToString;
      Values[RepeatTrials] := ARepeatTrials.ToString;
      Values[HasConsequence] := AHasConsequence.ToString;
    end;

    with MTSKeys do begin
      Values[Relation] := ARelation;
      Values[Samples] := ASamples.ToString;
      Values[Comparisons] := AComparisons.ToString;
      Values[Word] := AWord.Caption;

      for i := Low(AWord.Comparisons) to High(AWord.Comparisons) do begin
        case AWord.Phase.CompModality of
          ModalityA : LWord := AWord.Comparisons[i].Audio^;
          ModalityB : LWord := AWord.Comparisons[i].Image^;
          ModalityC : LWord := AWord.Comparisons[i].Text^;
          ModalityD : LWord := AWord.Comparisons[i].Speech^;
          ModalityNone : LWord := EmptyWord; // should never occur
        end;
        if not LWord.IsEmpty then begin
          Values[Comparison+(i+1).ToString] := LWord.Caption;
        end;
      end;
    end;
  end;
  Writer.WriteTrial;
end;

procedure WriteToConfigurationFile(AFilename : string);
var
  LTrialID : integer;
  LBlockID : integer;
  LCycle : integer;
  LBackUpBlock : integer = 0;
  LEndOnCriteria : Boolean;
  LTrials : integer;
  LHitCriterion : integer;
  LComparisons : integer;
  LCondition : integer;
  LBackUpBlockErrors : integer;
  LMaxBlockRepetition : integer;
  LMaxBlockRepetitionInSession, LNextBlockOnHitCriterion, i: integer;
  LName : string;
  LRelation : string;
  LCode , LInstruction: string;
  LHasConsequence , LEndOnHitCriterion: Boolean;
  LWord : TWord;
  LPhase : TPhase;
  LStartAt : TStartAt;
  LParser : TCSVRows;
  LRow : TStringList;

  function ToAlphaNumericCode(S : string) : TAlphaNumericCode;
  var
    LErrorCode : Word;
  begin
    Val(S, Result, LErrorCode);
    if LErrorCode <> 0 then
      Result := NA;
  end;

  function GetWord(APhase : TPhase; ACode : TAlphaNumericCode) : TWord;
  const
    LCodes = [Low(E1CyclesCodeRange)..High(E1CyclesCodeRange)];
  begin
    if ACode in LCodes then begin
      Result := HashWords[E1WordPerCycleCode[APhase.Cycle, ACode]]^;
      Result.Phase := APhase;
      SetComparisons(Result);
    end;
  end;

  function Validated : Boolean;
  begin
    Result := (LHitCriterion >= 0) and
              (LHitCriterion <= 100);
    if not Result then
      raise Exception.Create('Hit porcentage criterion is not valid: '+
        LHitCriterion.ToString);
  end;

begin
  //Format('%.2d', [LCycle])
  Writer := TConfigurationWriter.Create(ConfigurationFile);
  LParser := TCSVRows.Create;
  try
    // parse blocks
    if BlocksFileExists(AFilename) then begin
      LParser.Clear;
      LParser.LoadFromFile(InsideBlocksSubFolder(AFilename));
      for LRow in LParser do  begin
        with LRow, BlockKeys do begin
          LBlockID := Values['ID'].ToInteger -1;
          LBackUpBlock :=
            Values[NextBlockOnNotCriterionKey].ToInteger -1;
          LBackUpBlockErrors :=
            Values[BackUpBlockErrorsKey].ToInteger;
          LMaxBlockRepetition :=
            Values[MaxBlockRepetitionKey].ToInteger;
          LMaxBlockRepetitionInSession :=
            Values[MaxBlockRepetitionInSessionKey].ToInteger;
          LEndOnHitCriterion :=
            Values[EndSessionOnHitCriterionKey].ToBoolean;
          LNextBlockOnHitCriterion :=
            Values[NextBlockOnHitCriterionKey].ToInteger -1;
          LHitCriterion :=
            Values[CrtHitPorcentageKey].ToInteger;
        end;

        if not Validated then Exit;

        Writer.CurrentBlock := LBlockID;
        with Writer.BlockConfig, BlockKeys do begin
          Values['Name'] :=
            'Block ' + LBlockID.ToString;
          Values[NextBlockOnNotCriterionKey] :=
            LBackUpBlock.ToString;
          Values[BackUpBlockErrorsKey] :=
            LBackUpBlockErrors.ToString;
          Values[MaxBlockRepetitionKey] :=
            LMaxBlockRepetition.ToString;
          Values[MaxBlockRepetitionInSessionKey] :=
            LMaxBlockRepetitionInSession.ToString;
          Values[EndSessionOnHitCriterionKey] :=
            LEndOnHitCriterion.ToString;
          Values[NextBlockOnHitCriterionKey] :=
            LNextBlockOnHitCriterion.ToString;
          Values[CrtHitPorcentageKey] := LHitCriterion.ToString;
        end;
        Writer.WriteBlock;
      end;
    end else begin
      LBlockID := 0;
      Writer.CurrentBlock := LBlockID;
      with Writer.BlockConfig do begin
        Values['Name'] := 'Block ' + LBlockID.ToString;
      end;
      Writer.WriteBlock;
    end;

    // parse trials
    if BaseFileExists(AFilename) then begin
      LParser.Clear;
      LParser.LoadFromFile(InsideBaseFolder(AFilename));
      for LRow in LParser do  begin
        with LRow do begin
          LTrialID     := Values['ID'].ToInteger;
          LCycle       := Values['Cycle'].ToInteger;
          LCondition   := Values['Condition'].ToInteger;
          LBlockID     := Values['Block'].ToInteger -1;
          LTrials      := Values['Trials'].ToInteger; // TODO
          LComparisons := Values['Comparisons'].ToInteger;
          LRelation    := Values['Relation'];
          LCode        := Values['Code'];
          LHasConsequence := True;  // TODO
        end;
        LPhase := GetPhase(LCycle, LCondition, LRelation);
        LWord := GetWord(LPhase, ToAlphaNumericCode(LCode));

        Writer.CurrentBlock := LBlockID;
        for i := 0 to LTrials -1 do begin
          LName :=
            LWord.Caption + #32 + LRelation + #32 + LComparisons.ToString + 'C';
          WriteTrials(
            LName, LCode, LRelation, LWord, LComparisons, LHasConsequence);
        end;
      end;

      LStartAt.Block := 0;
      LStartAt.Trial:= 0;
      Writer.StartAt := LStartAt;
    end;

    // parse instructions to trials
    if InstructionsFileExist(AFilename) then begin
      LParser.Clear;
      LParser.LoadFromFile(InsideInstructionsSubFolder(AFilename));
       for LRow in LParser do  begin
        with LRow, TrialKeys do begin
          LBlockID     := Values['Block'].ToInteger-1;
          LTrialID     := Values['Trial'].ToInteger-1;
          LInstruction := Values[Instruction];
          Writer.WriteInstruction(LBlockID, LTrialID,
            Instruction, LInstruction);
        end;
      end;
    end;

  finally
    Writer.Free;
    LParser.Free;
  end;
end;

end.
