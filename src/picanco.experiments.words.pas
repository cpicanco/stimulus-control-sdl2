{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.words;

{$mode ObjFPC}{$H+}

interface

uses picanco.experiments.words.types;

var
  Words : TWords;
  NewWords : TWords;
  HashWords : THashWords;
  HashNewWords : THashWords;


procedure SetComparisons(var AWord: TWord);
function GetPhase(ACycle, ACondition : integer; ARelation: string) : TPhase;
function GetModalityFromLetter(ALetter : string) : TModality;
//procedure Initialize;
//procedure Finalize;

implementation

uses
  Classes, SysUtils, StrUtils
  , sdl.app.output
  , picanco.experiments.constants
  , picanco.experiments.words.constants;

procedure SetComparisons(var AWord: TWord);
var
  i: Integer;
  Code : TAlphaNumericCode;
  LWord : PTWord;
  LCandidateNegativeWords : TWordList;
  LCandidateNegativeComparisons : TWordList;
  LCandidateNegativeWordsWithNewImages : TWordList;
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
    for i := Low(AWord.CandidateNegativeWords) to
             High(AWord.CandidateNegativeWords) do begin
      LCandidateNegativeWords.Add(AWord.CandidateNegativeWords[i]);
    end;

    case AWord.CycleCode of
      T1 : begin
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R1]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R2]]);
      end;
      T2 : begin
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T1]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R1]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R2]]);
      end;
      R1 : begin
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T1]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R2]]);
      end;
      R2 : begin
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T1]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R1]]);
      end;

      A1, A2: begin
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T1]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, T2]]);
        LCandidateNegativeComparisons.Add(
          HashWords[E1WordPerCycleCode[AWord.Cycle, R1]]);
      end;
    end;

    for i := Low(E1WordsWithNewImages) to High(E1WordsWithNewImages) do begin
      LCandidateNegativeWordsWithNewImages.Add(
        HashNewWords[E1WordsWithNewImages[i]]);
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

  Print(Length(Words).ToString);
  for i := Low(Words) to High(Words) do begin
    SetNegativeComparisons(Words[i], Words);
    Print('');
    Print(Words[i].ToString);
  end;

  HashWords := THashWords.Create;
  for i := Low(Words) to High(Words) do begin
    HashWords[Words[i].Caption] := @Words[i];
  end;

  SetLength(NewWords, 0);
  for i := Low(E1WordsWithNewImages) to High(E1WordsWithNewImages) do begin
    SetLength(NewWords, Length(NewWords) + 1);
    NewWords[i].Caption := E1WordsWithNewImages[i];
    NewWords[i].Filenames := GetWordFilenames(E1WordsWithNewImages[i]);
  end;

  HashNewWords := THashWords.Create;
  for i := Low(NewWords) to High(NewWords) do begin
    HashNewWords[NewWords[i].Caption] := @NewWords[i];
  end;
end;

procedure Finalize;
begin
  Consonants.Free;
  Vowels.Free;
  HashWords.Free;
  HashNewWords.Free;
end;

initialization
  Initialize;

finalization
  Finalize;

end.

