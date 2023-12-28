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
      constructor Create; override;
      procedure AssignParameters(AParameters : TStringList); override;
      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses
  sdl.app.trials.mts
  , session.constants.trials
  , session.constants.trials.pseudowords
  , picanco.experiments.words
  , picanco.experiments.constants;

{ TCSVPseudowordsTrials }

// You are allowed to use "AfterLoadingParameters"
// to build dinamically created parameters.
procedure TCSVPseudowordsTrials.AfterLoadingParameters(Sender: TObject);
  function ToAlphaNumericCode(S : string) : TAlphaNumericCode;
  var
    LErrorCode : Word;
  begin
    Val(S, Result, LErrorCode);
    if LErrorCode <> 0 then
      Result := NA;
  end;

  function GetWord(APhase : TPhase; ACode : TAlphaNumericCode) : TWord;
  var
    LCode : string;
  begin
    case ACode of
      Low(E1PreTrainingRange)..High(E1PreTrainingRange): begin
        Result := HashPreTrainingWords[UniqueCodeToStr(ACode)]^;
      end;
      Low(E1CyclesCodeRange)..High(E1CyclesCodeRange): begin
       Result := HashWords[E1WordPerCycleCode[APhase.Cycle, ACode]]^;
      end;
      Low(E1WordsWithCodesRange)..High(E1WordsWithCodesRange): begin
       Result := HashWords[E1WordsWithCodes[ACode]]^;
      end;
     else begin
       WriteStr(LCode, ACode);
       raise Exception.Create('Unknown Word: '+ LCode);
     end;
    end;
    Result.Phase := APhase;
    //SetComparisons(Result);
  end;
begin
  inherited AfterLoadingParameters(Sender);
  FPhase := GetPhase(FCycle, FCondition, Relation);
  FWord := GetWord(FPhase, ToAlphaNumericCode(FCode));
  FName := TrialID.ToString + #32 + '(' + FWord.Caption + #32 +
      Relation + #32 + Comparisons.ToString + 'C)';
  FRefName := FCode;
end;

constructor TCSVPseudowordsTrials.Create;
begin
  inherited Create;
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
      { Instance }.AssignParameters;
      FWriter.WriteTrial;
    until Done;
  Do not use AssignParameters for building dinamically created parameters,
  unless you known what you are doing.
  You are allowed to assign not registered parameters to AParameters. }
procedure TCSVPseudowordsTrials.AssignParameters(AParameters: TStringList);
var
  LWord: TWord;
  i: Integer;
begin
  inherited AssignParameters(AParameters);
  with ParserTrialsPseudowordsMTS, AParameters do begin
    Values[SampleKey+IntToStr(1)] := FWord.Caption;
    SetComparisons(FWord);
    for i := Low(FWord.Comparisons) to High(FWord.Comparisons) do begin
      case FWord.Phase.CompModality of
        ModalityA : LWord := FWord.Comparisons[i].Audio^;
        ModalityB : LWord := FWord.Comparisons[i].Image^;
        ModalityC : LWord := FWord.Comparisons[i].Text^;
        ModalityD : LWord := FWord.Comparisons[i].Speech^;
        ModalityNone : LWord := EmptyWord; // should never occur
      end;
      if not LWord.IsEmpty then begin
        Values[ComparisonKey+(i+1).ToString] := LWord.Caption;
      end;
    end;
  end;
end;

end.


