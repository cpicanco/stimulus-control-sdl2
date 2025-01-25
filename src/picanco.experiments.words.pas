{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.words;

{$mode ObjFPC}{$H+}

interface

uses Generics.Collections, picanco.experiments.words.types;

type
  TCodes = specialize TList<TAlphaNumericCode>;

var
  PreTrainingWords : TWords;
  Words : TWords;
  NewWords : TWords;
  HashWords : THashWords;
  HashNewWords : THashWords;
  HashPreTrainingWords : THashWords;
  SessionCodes : TCodes;


procedure SetComparisons(var AWord: TWord; ALastWord: TWord);
function GetPhase(ACycle, ACondition : integer; ARelation: string) : TPhase;
function GetModalityFromLetter(ALetter : string) : TModality;
function GetWord(APhase : TPhase; ACode : TAlphaNumericCode) : TWord;
function ToAlphaNumericCode(S : string) : TAlphaNumericCode;
procedure Initialize;
procedure Finalize;

implementation

uses
  Classes, SysUtils, StrUtils
  //, sdl.app.output
  , picanco.experiments.constants
  , picanco.experiments.words.constants;

var
  TTFlip : Boolean = False;
  RRFlip : Boolean = False;
  AAFlip : Boolean = False;

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

var
  LLastPositiveCode : TAlphaNumericCode = NA;

procedure SetComparisons(var AWord: TWord; ALastWord: TWord);
var
  i: Integer;
  LCandidateNegativeWords : TWordList;
  LCandidateNegativeComparisons : TWordList;
  LCandidateNegativeWordsWithNewImages : TWordList;
  LCode2, LCode3: TAlphaNumericCode;
  //LM, LS : string;
begin
  for i := Low(AWord.Comparisons) to High(AWord.Comparisons) do begin
    with AWord.Comparisons[i] do begin
      Audio  := @EmptyWord;
      Image  := @EmptyWord;
      Text   := @EmptyWord;
      Speech := @EmptyWord;
    end;
  end;

  LCandidateNegativeWords := TWordList.Create;
  LCandidateNegativeComparisons := TWordList.Create;
  LCandidateNegativeWordsWithNewImages := TWordList.Create;
  try

    for i := Low(E1WordsWithNewImages) to High(E1WordsWithNewImages) do begin
      LCandidateNegativeWordsWithNewImages.Add(
        HashNewWords[E1WordsWithNewImages[i]]);
    end;

    for i := Low(AWord.CandidateNegativeWords) to
             High(AWord.CandidateNegativeWords) do begin
      LCandidateNegativeWords.Add(AWord.CandidateNegativeWords[i]);
    end;

    if AWord.CycleCode <> ALastWord.CycleCode then begin
      LLastPositiveCode := ALastWord.CycleCode;
    end;

    //WriteStr(LM, AWord.CycleCode);
    //WriteStr(LS, LLastPositiveCode);
    //Print(LM+'-'+LS);

    case AWord.CycleCode of
      T1 : begin
        TTFlip := not TTFlip;

        case LLastPositiveCode of
          R1: begin
            LCode2 := R1;
            if TTFlip then begin
              LCode3 := T2;
            end else begin
              LCode3 := R2;
            end;
          end;

          R2: begin
            LCode2 := R2;
            if TTFlip then begin
              LCode3 := R1;
            end else begin
              LCode3 := T2;
            end;
          end;

          otherwise begin
            LCode2 := T2;
            if TTFlip then begin
              LCode3 := R1;
            end else begin
              LCode3 := R2;
            end;
          end;
        end;
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
      end;

      T2 : begin
        TTFlip := not TTFlip;

        case LLastPositiveCode of
          R1: begin
            LCode2 := R1;
            if TTFlip then begin
              LCode3 := T1;
            end else begin
              LCode3 := R2;
            end;
          end;

          R2: begin
            LCode2 := R2;
            if TTFlip then begin
              LCode3 := R1;
            end else begin
              LCode3 := T1;
            end;
          end;

          otherwise begin
            LCode2 := T1;
            if TTFlip then begin
              LCode3 := R1;
            end else begin
              LCode3 := R2;
            end;
          end;
        end;
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
      end;

      R1 : begin
        RRFlip := not RRFlip;

        case LLastPositiveCode of
          T1: begin
            LCode2 := T1;
            if RRFlip then begin
              LCode3 := T2;
            end else begin
              LCode3 := R2;
            end;
          end;

          T2: begin
            LCode2 := T2;
            if RRFlip then begin
              LCode3 := R2;
            end else begin
              LCode3 := T1;
            end;
          end;

          otherwise begin
            LCode2 := R2;
            if RRFlip then begin
              LCode3 := T1;
            end else begin
              LCode3 := T2;
            end;
          end;
        end;
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
      end;

      R2 : begin
        RRFlip := not RRFlip;

        case LLastPositiveCode of
          T1: begin
            LCode2 := T1;
            if RRFlip then begin
              LCode3 := T2;
            end else begin
              LCode3 := R1;
            end;
          end;

          T2: begin
            LCode2 := T2;
            if RRFlip then begin
              LCode3 := R1;
            end else begin
              LCode3 := T1;
            end;
          end;

          otherwise begin
            LCode2 := R1;
            if RRFlip then begin
              LCode3 := T1;
            end else begin
              LCode3 := T2;
            end;
          end;
        end;
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
      end;

      A1, A2 : begin
        AAFlip := not AAFlip;
        if AAFlip then begin
          LCode2 := T1;
          LCode3 := T2;
        end else begin
          LCode2 := R1;
          LCode3 := R2;
        end;
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Phase.Cycle, LCode3]]);
      end;

      X1: begin
        LCandidateNegativeComparisons.Add(
          HashPreTrainingWords['X2']);
        LCandidateNegativeComparisons.Add(
          HashPreTrainingWords['Y1']);
        LCandidateNegativeComparisons.Add(
          HashPreTrainingWords['Y2']);
      end;

      X2: begin
        LCandidateNegativeComparisons.Add(
          HashPreTrainingWords['X1']);
        LCandidateNegativeComparisons.Add(
          HashPreTrainingWords['Y1']);
        LCandidateNegativeComparisons.Add(
          HashPreTrainingWords['Y2']);
      end;

      P1 : begin
        { no negative comparisons }
      end;

      P2 : begin
        { no negative comparisons }
      end;

      otherwise
        raise EArgumentOutOfRangeException.Create('SetComparisons error');
    end;

    for i := Low(AWord.Comparisons) to High(AWord.Comparisons) do begin
      with AWord.Comparisons[i] do begin
        Audio  := @EmptyWord;
        Image  := @EmptyWord;
        Text   := @EmptyWord;
        Speech := @EmptyWord;
        if i <= High(AWord.CandidateNegativeWords) then begin
          if i = 0 then begin
            case AWord.Phase.CompModality of
              ModalityA: Audio  := @AWord;
              ModalityB: Image  := @AWord;
              ModalityC: Text   := @AWord;
              ModalityD: Speech := @AWord;
              else
                raise Exception.Create(
                  'picanco.experiments.words.SetComparisons:'+
                  'Unknown modality in compararison '+ (i+1).ToString);
            end;
          end else begin
            case AWord.Phase.CompModality of
              ModalityA: Audio  := GetRandomWord(LCandidateNegativeWords);

              ModalityB: begin
                case AWord.Phase.Condition of
                  Condition_BC_CB_Testing:
                    Image :=
                      GetRandomWord(LCandidateNegativeWordsWithNewImages);
                  else
                    Image :=
                      GetNextComparison(LCandidateNegativeComparisons);
                end;
              end;

              ModalityC: Text   := GetRandomWord(LCandidateNegativeWords);
              ModalityD: Speech := @EmptyWord;
              else
                raise Exception.Create(
                  'picanco.experiments.words.SetComparisons:'+
                  'Unknown modality in compararison '+ (i+1).ToString);
            end;
          end;
        end;
      end;
    end;
  finally
    LCandidateNegativeWords.Free;
    LCandidateNegativeComparisons.Free;
    LCandidateNegativeWordsWithNewImages.Free;
  end;
end;

function GetPhase(ACycle, ACondition: integer; ARelation: string): TPhase;
begin
  Result.SampModality :=
    GetModalityFromLetter(ExtractDelimited(1, ARelation,['-']));
  Result.CompModality:=
    GetModalityFromLetter(ExtractDelimited(2, ARelation,['-']));

  case ACycle of
    0 : Result.Cycle := CycleNone;
    1 : Result.Cycle := Cycle1;
    2 : Result.Cycle := Cycle2;
    3 : Result.Cycle := Cycle3;
    4 : Result.Cycle := Cycle4;
    5 : Result.Cycle := Cycle5;
    6 : Result.Cycle := Cycle6;
  end;

  case ACondition of
    0 : Result.Condition := ConditionNone;
    1 : Result.Condition := Condition_AB;
    2 : Result.Condition := Condition_AC_CD;
    3 : Result.Condition := Condition_BC_CB_Training;
    4 : Result.Condition := Condition_BC_CB_Testing;
    5 : Result.Condition := Condition_CD_1;
    6 : Result.Condition := Condition_AC;
    7 : Result.Condition := Condition_CD_2;
    8 : Result.Condition := Condition_CD_3;
  end;

  case Result.Condition of
    ConditionNone   : Result.Stage:= StageNone;
    Condition_AB,
    Condition_AC_CD : Result.Stage:= StageTraining;
    else
      Result.Stage:= StageTesting;
  end;
end;

function GetModalityFromLetter(ALetter: string): TModality;
begin
  case ALetter of
    'A' : Result := ModalityA;
    'B' : Result := ModalityB;
    'C' : Result := ModalityC;
    'D' : Result := ModalityD;
    else
      raise Exception.Create('Unknown modality: '+ ALetter);
  end;
end;

procedure Initialize;
var
  i, j : Integer;
begin
  SessionCodes := TCodes.Create;

  EmptyWord.Caption := '----';
  EmptyWord.Filenames.Audio:='--Empty--';
  EmptyWord.Filenames.Image:='--Empty--';
  EmptyWord.Filenames.Text:='--Empty--';
  EmptyWord.Filenames.Speech:='--Empty--';
  EmptyWord.Syllable1.Consonant.Ord := csNone;
  EmptyWord.Syllable1.Vowel.Ord := vsNone;
  EmptyWord.Syllable2.Consonant.Ord := csNone;
  EmptyWord.Syllable2.Vowel.Ord := vsNone;

  Consonants := TConsonants.Create;
  Consonants.Add(PlosiveBilabial);
  Consonants.Add(NonSibilantFricative);
  Consonants.Add(LateralApproximantAlveolar);
  Consonants.Add(NasalAlveolar);

  Vowels := TVowels.Create;
  Vowels.Add(OpenFront);
  Vowels.Add(OpenMidFront);
  Vowels.Add(CloseFront);
  Vowels.Add(OpenMidBack);

  SetLength(Syllables, Consonants.Count * Vowels.Count);
  for i := 0 to Consonants.Count -1 do
    for j := 0 to Vowels.Count-1 do
    begin
      Syllables[i * Consonants.Count + j].Consonant := Consonants[i];
      Syllables[i * Vowels.Count + j].Vowel := Vowels[j];
    end;

  SetLength(Words, 0);
  for i := Low(Syllables) to High(Syllables) do
    for j := Low(Syllables) to High(Syllables) do
    begin
      //if Syllables[i] = Syllables[j] then
      //  Continue;

      //if Syllables[i].Consonant = Syllables[j].Consonant then
      //  Continue;
      //
      //if Syllables[i].Vowel = Syllables[j].Vowel then
      //  Continue;

      SetLength(Words, Length(Words) + 1);
      Words[Length(Words) - 1].Syllable1 := Syllables[i];
      Words[Length(Words) - 1].Syllable2 := Syllables[j];
    end;

  for i := Low(Words) to High(Words) do begin
    InitializeWord(Words[i]);
  end;

  //Print(Length(Words).ToString);
  for i := Low(Words) to High(Words) do begin
    SetNegativeComparisons(Words[i], Words);
    //Print('');
    //Print(Words[i].ToString);
  end;

  HashWords := THashWords.Create;
  for i := Low(Words) to High(Words) do begin
    HashWords.Add(Words[i].Caption, @Words[i]);
  end;

  SetLength(NewWords, 0);
  for i := Low(E1WordsWithNewImages) to High(E1WordsWithNewImages) do begin
    SetLength(NewWords, Length(NewWords) + 1);
    NewWords[i].Caption := E1WordsWithNewImages[i];
    NewWords[i].Filenames := GetWordFilenames(E1WordsWithNewImages[i]);
  end;

  HashNewWords := THashWords.Create;
  for i := Low(NewWords) to High(NewWords) do begin
    HashNewWords.Add(NewWords[i].Caption, @NewWords[i]);
  end;

  SetLength(PreTrainingWords, Length(E1UniqueCodesPreTraining));
  HashPreTrainingWords := THashWords.Create;
  for i := Low(E1UniqueCodesPreTraining) to
           High(E1UniqueCodesPreTraining) do begin
    PreTrainingWords[i].Caption := UniqueCodeToStr(E1UniqueCodesPreTraining[i]);
    PreTrainingWords[i].CycleCode := E1UniqueCodesPreTraining[i];
    SetLength(PreTrainingWords[i].CandidateNegativeWords, MaxComparisons-1);
    HashPreTrainingWords.Add(PreTrainingWords[i].Caption, @PreTrainingWords[i]);
  end;
end;

procedure Finalize;
begin
  Consonants.Free;
  Vowels.Free;
  HashWords.Free;
  HashNewWords.Free;
  HashPreTrainingWords.Free;
  SessionCodes.Free;
end;

end.

