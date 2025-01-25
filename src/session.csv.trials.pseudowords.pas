{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials.pseudowords;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , session.csv.trials.mts
  , picanco.experiments.words.types;

type

  { TCSVPseudowordsTrials }

  TCSVPseudowordsTrials = class(TCSVTrialsMTS)
    private // registered parameters
      FCondition   : integer;
      FCode        : string;
      FName        : string;
      FRefName     : string;
    private // complex dinamically created parameters
      FWord : TWord;
      FPhase : TPhase;
    protected
      FCycle       : integer;
      procedure AfterLoadingParameters(Sender: TObject); override;
    public
      constructor Create(ASource: string); override;
      procedure AssignParameters(AParameters : TStringList); override;
      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses
  session.constants.trials
  , session.constants.trials.pseudowords
  , picanco.experiments.words
  ;

{ TCSVPseudowordsTrials }

// You are allowed to use "AfterLoadingParameters"
// to build dinamically created parameters.
procedure TCSVPseudowordsTrials.AfterLoadingParameters(Sender: TObject);
begin
  inherited AfterLoadingParameters(Sender);
  FPhase := GetPhase(FCycle, FCondition, Relation);
  FWord := GetWord(FPhase, ToAlphaNumericCode(FCode));
  FName :=
    TrialID.ToString + #32 +
    'Cycle ' + FCycle.ToString + #32 +
    FWord.Caption + #32 +
    Relation + #32 +
    Comparisons.ToString + 'C';

  if (TrialID = 109) or (TrialID = 110) then begin
    FRefName := FCode+'-'+Relation+'-known';
  end else begin
    FRefName := FCode+'-'+Relation;
  end;
end;

constructor TCSVPseudowordsTrials.Create(ASource: string);
begin
  inherited Create(ASource);
  FCycle       := 0;
  FCondition   := 0;
  FCode        := '';
  FName        := '';
  FRefName     := '';

  with ParserTrialsPseudowordsMTS do begin
    RegisterParameter(CycleKey,
      @FCycle, FCycle);
    RegisterParameter(ConditionKey,
      @FCondition, FCondition);
    RegisterParameter(CodeKey,
      @FCode, FCode);
    RegisterParameter(NameKey,
      @FName, FName);
    RegisterParameter(ReferenceNameKey,
      @FRefName, FRefName);
  end;
end;

{ You may call AssignParameters many times after loading parameters.
  For example, in pseudocode:
    repeat
      % Instance %.AssignParameters;
      FWriter.WriteTrial;
    until Done;

  AssignParameters occurs before trial ordering,
  so do not use it for building dinamically created parameters,
  unless you known what you are doing.
  You are allowed to assign not registered parameters to AParameters. }
procedure TCSVPseudowordsTrials.AssignParameters(AParameters: TStringList);
//var
//  LWord: TWord;
//  i: Integer;
begin
  inherited AssignParameters(AParameters);
  //with ParserTrialsPseudowordsMTS, AParameters do begin
  //  Values[SampleKey+IntToStr(1)] := FWord.Caption;
  //  SetComparisons(FWord, SessionCodes);
  //  for i := Low(FWord.Comparisons) to High(FWord.Comparisons) do begin
  //    case FWord.Phase.CompModality of
  //      ModalityA : LWord := FWord.Comparisons[i].Audio^;
  //      ModalityB : LWord := FWord.Comparisons[i].Image^;
  //      ModalityC : LWord := FWord.Comparisons[i].Text^;
  //      ModalityD : LWord := FWord.Comparisons[i].Speech^;
  //      ModalityNone : LWord := EmptyWord; // should never occur
  //    end;
  //    if not LWord.IsEmpty then begin
  //      Values[ComparisonKey+(i+1).ToString] := LWord.Caption;
  //    end;
  //  end;
  //end;
end;

end.


