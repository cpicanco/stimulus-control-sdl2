{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.configuration;

{$Mode objfpc}{$H+}
{$ModeSwitch AdvancedRecords}
{$ModeSwitch TypeHelpers}

interface

uses Classes, SysUtils;

type

  TBlockRepeatStyle = (None, Consecutive, Global, ConsecutiveAndGlobal);
  TBlockRepeatStyleRange = TBlockRepeatStyle.None..TBlockRepeatStyle.ConsecutiveAndGlobal;

  TBlockEndCriterionStyle = (HitCount, MissCount, ConsecutiveHits, ConsecutiveMisses, HitPorcentage, MissPorcentage);
  TBlockEndCriterionStyleRange = TBlockEndCriterionStyle.HitCount..TBlockEndCriterionStyle.MissPorcentage;

  TBlockEndCriterionEvaluationTime = (OnTrialEnd, OnBlockEnd);
  TBlockEndCriterionEvaluationTimeRange = TBlockEndCriterionEvaluationTime.OnTrialEnd..TBlockEndCriterionEvaluationTime.OnBlockEnd;

  { TTrialConfiguration }

  TTrialConfiguration = record
    ID : integer;
    Kind: string;
    ReferenceName: string;
    Parameters: TStringList;
    class operator = (A, B: TTrialConfiguration): Boolean;
    function ToData : string;
  end;

  TTrials = array of TTrialConfiguration;

  { TBlockConfiguration }

  TBlockConfiguration = record
    ID : integer;
    Name: string;
    TotalTrials: integer;
    EndSessionOnCriterion	: Boolean;
    EndSessionOnNotCriterionAfterBlockRepetitions : Boolean;

    RepeatStyle : TBlockRepeatStyle;
    EndCriterionStyle : TBlockEndCriterionStyle;
    EndCriterionEvaluationTime : TBlockEndCriterionEvaluationTime;

    MaxBlockRepetitionConsecutives : integer;
    MaxBlockRepetitionInSession	: integer;

    NextBlockOnCriterion : integer;
    NextBlockOnNotCriterion : integer;
    EndCriterionValue : integer;
    Reinforcement : integer;

    //Counter : string;
    //AutoEndSession : Boolean;
    //MaxCorrection: integer;
    //BkGnd: integer;
    //ITI: integer;
    //DefNextBlock: string;
    //CrtConsecutive: integer;
    //CrtHitValue: integer;
    //CrtConsecutiveHit: integer;
    //CrtConsecutiveHitPerType : integer;
    //CrtConsecutiveMiss : integer;
    //CrtMaxTrials : integer;
    //CrtCsqHit : integer;
    Trials: TTrials;
    class operator = (A, B: TBlockConfiguration): Boolean;
    function ToData : string;
  end;

  TBlocks = array of TBlockConfiguration;

  { TBlockRepeatStyleHelper }

  TBlockRepeatStyleHelper = type helper for TBlockRepeatStyle
    function ToString: string;
  end;

  { TBlockEndCriterionStyleHelper }

  TBlockEndCriterionStyleHelper = type helper for TBlockEndCriterionStyle
    function ToString: string;
  end;

  { TBlockEndCriterionEvaluationTimeHelper }

  TBlockEndCriterionEvaluationTimeHelper = type helper for TBlockEndCriterionEvaluationTime
    function ToString: string;
  end;

  { TBlockStringHelper }

  TBlockStringHelper = type helper(TStringHelper) for string
    function ToRepeatStyle : TBlockRepeatStyle;
    function ToEndCriterionStyle : TBlockEndCriterionStyle;
    function ToEndCriterionEvaluationTime : TBlockEndCriterionEvaluationTime;
  end;

implementation

function AsNameValue(AName, AValue: string):string;
begin
  Result := String.Join('=', [AName, AValue]);
end;

{ TTrialConfiguration }

class operator TTrialConfiguration.=(A, B: TTrialConfiguration): Boolean;
begin
  Result := A.ID = B.ID;
end;

function TTrialConfiguration.ToData: string;
begin
  Result := String.Join(LineEnding,[
    AsNameValue('ID', ID.ToString),
    AsNameValue('Kind', Kind),
    AsNameValue('ReferenceName', ReferenceName),
    'Parameters:',
    Parameters.Text]);
end;

{ TBlockConfiguration }

class operator TBlockConfiguration.=(A, B: TBlockConfiguration): Boolean;
begin
  Result := A.ID = B.ID;
end;

function TBlockConfiguration.ToData: string;
begin
  Result := String.Join(LineEnding,[
    AsNameValue('ID', ID.ToString),
    AsNameValue('Name', Name),
    AsNameValue('TotalTrials', TotalTrials.ToString),
    AsNameValue('EndSessionOnCriterion', BoolToStr(EndSessionOnCriterion, True)),
    AsNameValue('EndSessionOnNotCriterionAfterBlockRepetitions',
      BoolToStr(EndSessionOnNotCriterionAfterBlockRepetitions, True)),

    AsNameValue('RepeatStyle', RepeatStyle.ToString),
    AsNameValue('EndCriterionStyle', EndCriterionStyle.ToString),
    AsNameValue('EndCriterionEvaluationTime',
      EndCriterionEvaluationTime.ToString),

    AsNameValue('MaxBlockRepetitionConsecutives',
      MaxBlockRepetitionConsecutives.ToString),

    AsNameValue('MaxBlockRepetitionInSession',
      MaxBlockRepetitionInSession.ToString),

    AsNameValue('NextBlockOnCriterion', NextBlockOnCriterion.ToString),
    AsNameValue('NextBlockOnNotCriterion', NextBlockOnNotCriterion.ToString),
    AsNameValue('EndCriterionValue', EndCriterionValue.ToString),
    AsNameValue('Reinforcement', Reinforcement.ToString),

    //AsNameValue('Counter', Counter),
    //AsNameValue('AutoEndSession', AutoEndSession.ToString),
    //AsNameValue('MaxCorrection', MaxCorrection.ToString),
    //AsNameValue('BackgroundColor', BkGnd.ToString),
    //AsNameValue('InterTrialInterval', ITI.ToString),
    //AsNameValue(DefNextBlock: string;
    //AsNameValue(CrtConsecutive: integer;
    //AsNameValue(CrtHitValue: integer;
    //AsNameValue(CrtConsecutiveHit: integer;
    //AsNameValue(CrtConsecutiveHitPerType : integer;
    //AsNameValue(CrtConsecutiveMiss : integer;
    //AsNameValue(CrtMaxTrials : integer;
    //AsNameValue(CrtCsqHit : integer;
    AsNameValue('TrialsLength', Length(Trials).ToString)]);
end;

{ TBlockRepeatStyleHelper }

function TBlockRepeatStyleHelper.ToString: string;
begin
  WriteStr(Result, Self);
end;

{ TBlockEndCriterionStyleHelper }

function TBlockEndCriterionStyleHelper.ToString: string;
begin
  WriteStr(Result, Self);
end;

{ TBlockEndCriterionEvaluationTimeHelper }

function TBlockEndCriterionEvaluationTimeHelper.ToString: string;
begin
  WriteStr(Result, Self);
end;

{ TBlockStringHelper }

function TBlockStringHelper.ToRepeatStyle: TBlockRepeatStyle;
var
  LValue : string = '';
begin
  for Result in TBlockRepeatStyleRange do begin
    WriteStr(LValue, Result);
    if LValue=Self then begin
      Exit;
    end;
  end;
  raise Exception.CreateFmt('TBlockRepeatStyle %s not found', [Self]);
end;

function TBlockStringHelper.ToEndCriterionStyle: TBlockEndCriterionStyle;
var
  LValue : string = '';
begin
  for Result in TBlockEndCriterionStyleRange do begin
    WriteStr(LValue, Result);
    if LValue=Self then begin
      Exit;
    end;
  end;
  raise Exception.CreateFmt('TBlockEndCriterionStyle %s not found', [Self]);
end;

function TBlockStringHelper.ToEndCriterionEvaluationTime: TBlockEndCriterionEvaluationTime;
var
  LValue : string = '';
begin
  for Result in TBlockEndCriterionEvaluationTimeRange do begin
    WriteStr(LValue, Result);
    if LValue=Self then begin
      Exit;
    end;
  end;
  raise Exception.CreateFmt('TBlockEndCriterionEvaluationTimeRange %s not found', [Self]);
end;

end.







