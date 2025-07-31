{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.words.types;

{$mode ObjFPC}{$H+}
{$modeSwitch advancedRecords}

interface

uses Generics.Collections;

const
  MaxComparisons = 4;

type
  TCycle = (CycleNone, Cycle1, Cycle2, Cycle3, Cycle4, Cycle5, Cycle6);

  TStage = (StageNone, StageTraining, StageTesting);

  TModality = (ModalityNone, ModalityA, ModalityB, ModalityC, ModalityD);

  TCondition = (
    ConditionNone,            // Pre-training
    Condition_AB,             // Audio-To-Picture
    Condition_AC_CD,          // Audio-To-Text   and Text-To-Speech
    Condition_BC_CB_Training, // Picture-To-Text and Text-To-Picture
    Condition_BC_CB_Testing,  // Picture-To-Text and Text-To-Picture
    Condition_CD_1,           // Text-To-Speech
    Condition_AC,             // Audio-To-Text
    Condition_CD_2,            // Text-To-Speech
    Condition_CD_3);          // Text-To-Speech

  TPhase = record
    Cycle : TCycle;
    Stage : TStage;
    Condition : TCondition;
    SampModality : TModality;
    CompModality : TModality;
  end;

  TOverlaps = (
    DifferentConsonantsAndVowels,
    EqualConsonants,
    EqualVowels,
    EqualConsonantsAndVowels);

  TConsonantOrd = (
    csNone,
    csPlosiveBilabial,
    csNonSibilantFricative,
    csLateralApproximantAlveolar,
    csNasalAlveolar);

  TVowelOrd = (
    vsNone,
    vsOpenFront,
    vsOpenMidFront,
    vsCloseFront,
    vsOpenMidBack);

  { TConsonant }

  TConsonant = record
    Ord: TConsonantOrd;
    // https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-ssml-phonetic-sets
    IPA: string;  // International Phonetic Alphabet
    SYM: string;  // Symbolic and Numerical Representation
    HumanReadable: string;
    procedure Next;
    class operator = (AConsonant1, AConsonant2: TConsonant): boolean;
  end;

  TConsonants = specialize TList<TConsonant>;

  { TVowel }

  TVowel = record
    Ord: TVowelOrd;
    IPA: string;  // International Phonetic Alphabet
    SYM: string;  // Symbolic and Numerical Representation
    HumanReadable: string;
    HumanReadableStress: string;
    procedure Next;
    class operator = (AVowel1, AVowel2: TVowel): boolean;
  end;

  TVowels = specialize TList<TVowel>;

  { TSyllable }

  TSyllable = record
    Consonant: TConsonant;
    Vowel: TVowel;
    class operator = (ASyllable1, ASyllable2: TSyllable): boolean;
  end;

  TSyllables = array of TSyllable;

  TWordFilenames = record
    Audio  : string;   // A
    Image  : string;   // B
    Text   : string;   // C
    Speech : string;   // D
  end;

  TAlphaNumericCode =
   (NA,
    X1,  X2,  Y1,  Y2,  P1, P2,  // Unique codes per shape, pre-teaching
    T1,  T2,  R1,  R2,  A1,  A2,     // Cycle codes
    T01, T02, T03, T04, T05, T06,    // Unique codes per word, teaching/testing
    T07, T08, T09, T10, T11, T12,
    R01, R02,
    A01, A02, A03, A04, A05, A06,
    A07, A08, A09, A10, A11, A12,
    A13, A14, A15, A16);

  TAlphaNumericCodes = array of TAlphaNumericCode;

  E1PreTrainingRange = X1..P2;
  E1CyclesRange = Cycle1..Cycle6;
  E1CyclesCodeRange = T1..A2;
  E1CyclesCodeRangeWithImages = T1..R2;
  E1WordsWithImagesRange = T01..R02;
  E1WordsWithCodesRange = T01..A16;

  // Reserved
  E1ReserUniqueCodeRange = R01..R02;

  // Teaching
  E1TeachUniqueCodeRange = T01..T12;

  // Probes
  E1ProbeUniqueCodeRange = A01..A12;

  { TWord }

  PTWord = ^TWord;
  TWord = record
    Syllable1 : TSyllable;
    Syllable2 : TSyllable;
    Cycle : TCycle;
    CycleCode  : TAlphaNumericCode;
    UniqueCode : TAlphaNumericCode;
    Code  : TAlphaNumericCode;
    Phase : TPhase;
    Overlaps : TOverlaps;
    Caption : string;
    Filenames : TWordFilenames;
    CandidateNegativeWords : array of PTWord;
    Comparisons : array [0..MaxComparisons] of record
      Audio : PTWord;
      Image : PTWord;
      Text  : PTWord;
      Speech : PTWord;
    end;
    function ToIPA : string;
    function ToSym : string;
    function ToHumanReadableText(AHasStress: Boolean = False): string;
    function ToString : string;
    function IsEmpty : Boolean;
  end;

  TWords = array of TWord;
  PTWords = array of PTWord;
  TWordList = specialize TList<PTWord>;
  THashWords = specialize TDictionary<string, PTWord>;

  procedure InitializeWord(var AWord: TWord);
  //procedure ShuffleWords(AWords: TWordList);
  function GetRandomWord(var AWords: TWordList): PTWord;
  function GetNextComparison(var AWords: TWordList): PTWord;
  function GetWordFilenames(ACaption: string) : TWordFilenames;
  procedure SetNegativeComparisons(var AWord: TWord; var AWords: TWords);

var
  Consonants : TConsonants;
  Vowels : TVowels;
  Syllables : TSyllables;
  EmptyWord : TWord;

implementation

uses Classes, SysUtils, picanco.experiments.constants
  , picanco.experiments.words.constants
  ;

//procedure ShuffleWords(AWords: TWordList);
//begin
//  with AWords do begin
//    for i := 0 to Count-1 do begin
//      r := Random(Count);
//      Exchange(i, r);
//    end;
//  end;
//end;

function GetRandomWord(var AWords: TWordList): PTWord;
var
  i : integer;
begin
  i := Random(AWords.Count);
  Result := AWords[i];
  AWords.Delete(i);
end;

function GetNextComparison(var AWords: TWordList): PTWord;
begin
  Result := AWords[0];
  AWords.Delete(0);
end;

procedure GetWordsFromLetter(AWord: TWord; AIndex: Integer;
  ACandidateWords: TWordList; var AWords: TWords);
var
  i : Integer;
  function OtherLettersAreDifferent(AIndex: integer;
    AWord1: TWord; AWord2: TWord; AFilter: array of integer):Boolean;
  var
    i : integer;
    LWord1 : TStringList;
    LWord2 : TStringList;
    AllChars             : TStringList;
  begin
    Result := True;
    LWord1 := TStringList.Create;
    LWord1.Sorted := True;
    LWord1.Duplicates:= dupIgnore;

    LWord2 := TStringList.Create;
    LWord2.Sorted := True;
    LWord2.Duplicates:= dupIgnore;

    AllChars := TStringList.Create;
    AllChars.Sorted := True;
    AllChars.Duplicates := dupIgnore;
    try
      for i in AFilter do begin
        if i = AIndex then Continue;
        LWord1.Append(AWord1.Caption[i]);
      end;

      for i in AFilter do begin
        if i = AIndex then Continue;
        LWord2.Append(AWord2.Caption[i]);
      end;

      for i := 0 to LWord1.Count -1 do begin
        AllChars.Append(LWord1[i]);
      end;
      for i := 0 to LWord2.Count -1 do begin
        AllChars.Append(LWord2[i]);
      end;

      if AllChars.Count < LWord1.Count + LWord2.Count then
        Result := False;
    finally
      AllChars.Free;
      LWord1.Free;
      LWord2.Free;
    end;
  end;

begin
  for i := Low(AWords) to High(AWords) do begin
    if (AWord.Caption <> AWords[i].Caption) and
       (AWord.Caption[AIndex] = AWords[i].Caption[AIndex]) then begin
      if AWord.Overlaps = AWords[i].Overlaps then
        case AWord.Overlaps of
          EqualVowels: begin
            if OtherLettersAreDifferent(
              AIndex, AWord, AWords[i], [1, 3]) then begin // 1,3 Consonants
               ACandidateWords.Add(@AWords[i]);
            end;
          end;
          EqualConsonants: begin
            if OtherLettersAreDifferent(
              AIndex, AWord, AWords[i], [2, 4]) then begin // 2, 4 vowels
               ACandidateWords.Add(@AWords[i]);
            end;
          end;
          EqualConsonantsAndVowels: begin
            ACandidateWords.Add(@AWords[i]);
          end;
          DifferentConsonantsAndVowels: begin
            if OtherLettersAreDifferent(
              AIndex, AWord, AWords[i], [1, 2, 3, 4]) then begin
              ACandidateWords.Add(@AWords[i]);
            end;
          end;
        end;
    end;
  end;
end;

function GetWordFilenames(ACaption: string) : TWordFilenames;
begin
  //Result.Audio  := Pool.MediaRootBasePath+DirectorySeparator+ACaption+'.wav';
  //Result.Image  := Pool.MediaRootBasePath+DirectorySeparator+ACaption+'.jpg';
  //Result.Text   := Pool.MediaRootBasePath+DirectorySeparator+ACaption+'.txt';
  //Result.Speech := Pool.MediaRootBasePath+DirectorySeparator+ACaption+'-spoken';
  Result.Audio  := ACaption;
  Result.Image  := ACaption;
  Result.Text   := ACaption;
  Result.Speech := ACaption;
end;

procedure SetNegativeComparisons(var AWord: TWord; var AWords: TWords);
var
  i: Integer;
  LNegativeComparisons : PTWords;
  LCandidateWords      : TWordList;
  LRandomWord          : PTWord;
  LIsUnique: Boolean;
  function UniqueComparison(AWord : PTWord; AWords : PTWords) : Boolean;
  var
    i : integer;
  begin
    Result := False;
    for i := Low(AWords) to High(AWords) do begin
      if AWord^.Caption = AWords[i]^.Caption then begin
         Exit;
      end;
    end;
    Result := True;
  end;

begin
  LNegativeComparisons := Default(PTWords);
  SetLength(LNegativeComparisons, Length(AWord.Caption));
  for i := Low(LNegativeComparisons) to High(LNegativeComparisons) do begin
    LNegativeComparisons[i] := @EmptyWord;
  end;

  LCandidateWords := TWordList.Create;
  try
    for i := 1 to Length(AWord.Caption) do
    begin
      LCandidateWords.Clear;
      GetWordsFromLetter(AWord, i, LCandidateWords, AWords);
      //Print(LCandidateWords.Count.ToString);
      if LCandidateWords.Count > 0 then begin
        repeat
          LRandomWord := GetRandomWord(LCandidateWords);
          LIsUnique := UniqueComparison(LRandomWord, LNegativeComparisons);
        until LIsUnique or (LCandidateWords.Count = 0);
        if LIsUnique then
          LNegativeComparisons[i - 1] := LRandomWord;
      end;
    end;
  finally
    LCandidateWords.Free;
  end;
  AWord.CandidateNegativeWords := LNegativeComparisons;
end;

procedure InitializeWord(var AWord: TWord);
begin
  AWord.Caption := AWord.ToHumanReadableText;
  AWord.Filenames := GetWordFilenames(AWord.Caption);
  AWord.CycleCode := CycleCodeFromWord(AWord.Caption, AWord.Cycle);
  AWord.UniqueCode := UniqueCodeFromWord(AWord.Caption);

  AWord.Overlaps := DifferentConsonantsAndVowels;
  if AWord.Syllable1 = AWord.Syllable2 then begin
    AWord.Overlaps := EqualConsonantsAndVowels;
    Exit;
  end;

  if AWord.Syllable1.Consonant = AWord.Syllable2.Consonant then begin
    AWord.Overlaps := EqualConsonants;
    Exit;
  end;

  if AWord.Syllable1.Vowel = AWord.Syllable2.Vowel then begin
    AWord.Overlaps := EqualVowels;
  end;
end;

{ TConsonant }

procedure TConsonant.Next;
begin
  case Self.Ord of
    csNone : Exit;
    csPlosiveBilabial: Self := NonSibilantFricative;
    csNonSibilantFricative: Self := LateralApproximantAlveolar;
    csLateralApproximantAlveolar: Self := NasalAlveolar;
    csNasalAlveolar: Self := PlosiveBilabial;
  end;
end;

class operator TConsonant.=(AConsonant1, AConsonant2: TConsonant): boolean;
begin
  Result := AConsonant1.HumanReadable = AConsonant2.HumanReadable;
end;

{ TVowel }

procedure TVowel.Next;
begin
  case Self.Ord of
    vsNone : Exit;
    vsOpenFront: Self := OpenMidFront;
    vsOpenMidFront: Self := CloseFront;
    vsCloseFront: Self := OpenMidBack;
    vsOpenMidBack: Self := OpenFront;
  end;
end;

class operator TVowel.=(AVowel1, AVowel2: TVowel): boolean;
begin
  Result := AVowel1.HumanReadable = AVowel2.HumanReadable;
end;

{ TSyllable }

class operator TSyllable.=(ASyllable1, ASyllable2: TSyllable): boolean;
begin
  Result :=
    (ASyllable1.Consonant.HumanReadable = ASyllable2.Consonant.HumanReadable) and
    (ASyllable1.Vowel.HumanReadable = ASyllable2.Vowel.HumanReadable);
end;

{ TWord }

function TWord.ToIPA: string;
begin
  Result :=
    Syllable1.Consonant.IPA + Syllable1.Vowel.IPA + '.ˈ'+
    Syllable2.Consonant.IPA + Syllable2.Vowel.IPA;
end;

function TWord.ToSym: string;
begin
  Result :=
    Syllable1.Consonant.SYM + #32 + Syllable1.Vowel.SYM + #32 + '-' + #32 +
    Syllable2.Consonant.SYM + #32 + Syllable2.Vowel.SYM + '1';
end;

function TWord.ToHumanReadableText(AHasStress: Boolean): string;
begin
  if AHasStress then begin
    Result :=
      Syllable1.Consonant.HumanReadable + Syllable1.Vowel.HumanReadable +
      Syllable2.Consonant.HumanReadable + Syllable2.Vowel.HumanReadableStress
  end else begin
    Result :=
      Syllable1.Consonant.HumanReadable + Syllable1.Vowel.HumanReadable +
      Syllable2.Consonant.HumanReadable + Syllable2.Vowel.HumanReadable;
  end;
end;

function TWord.ToString: string;
var
  LOverlaps : string;
begin
  WriteStr(LOverlaps, Overlaps);
  Result :=
    'Word: ' + Caption + LineEnding +
    'IPA: ' + ToIPA + LineEnding +
    'Overlaps:' + LOverlaps + LineEnding +
    'S-''s: '+ CandidateNegativeWords[0]^.Caption + #32 +
               CandidateNegativeWords[1]^.Caption + #32 +
               CandidateNegativeWords[2]^.Caption + #32 +
               CandidateNegativeWords[3]^.Caption;
end;

function TWord.IsEmpty: Boolean;
begin
  Result := Self.Caption = EmptyWord.Caption;
end;

end.

