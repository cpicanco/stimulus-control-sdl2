{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.dynamics.base;

{$mode ObjFPC}{$H+}

interface

uses
  Classes;

procedure SetTrialDynamics(AParameters: TStringList);

implementation

uses
  SysUtils,
  session.strutils,
  session.pool,
  session.constants.trials.dragdrop,
  session.constants.trials.pseudowords,
  session.constants.trials,
  picanco.experiments.words.types,
  picanco.experiments.words;

var
  LastWord : TWord;

procedure SetTrialDynamics(AParameters: TStringList);
var
  LRelation    , LStimuliFolder: string;
  LTrialID     : integer;
  LCycle       : integer;
  LCondition   : integer;
  LCode        : TAlphaNumericCode;
  LPhase       : TPhase;
  LWord        : TWord;
  LTmpWord     : TWord;
  i: Integer;
begin
  with AParameters do begin
    with ParserTrialsSourceKeys, ParserTrialsPseudowordsMTS do begin
      if Values[TrialIDSourceKey].Contains('mts-pseudowords') or
         Values[TrialIDSourceKey].Contains('mts-images') then begin

        LTrialID := Values['ID'].ToInteger;
        LCycle := Values[CycleKey].ToInteger;
        LCondition := Values[ConditionKey].ToInteger;
        LRelation := Values[RelationKey];
        LCode := ToAlphaNumericCode(Values[CodeKey]);

        LPhase := GetPhase(LCycle, LCondition, LRelation);
        LWord := GetWord(LPhase, LCode);

        // sample
        Values[SampleKey+IntToStr(1)] := LWord.Caption;

        // comparisons
        if LTrialID = 0 then begin
          SessionCodes.Clear;
        end;
        SetComparisons(LWord, LastWord);
        LastWord := LWord;
        for i := Low(LWord.Comparisons) to High(LWord.Comparisons) do begin
          case LWord.Phase.CompModality of
            ModalityA : LTmpWord := LWord.Comparisons[i].Audio^;
            ModalityB : LTmpWord := LWord.Comparisons[i].Image^;
            ModalityC : LTmpWord := LWord.Comparisons[i].Text^;
            ModalityD : LTmpWord := LWord.Comparisons[i].Speech^;
            ModalityNone : LTmpWord := EmptyWord; // should never occur
          end;

          if not LTmpWord.IsEmpty then begin
            Values[ComparisonKey+(i+1).ToString] := LTmpWord.Caption;
          end;
        end;
      end;
    end;

    with ParserTrialsSourceKeys, DragDropKeys do begin
      if Values[TrialIDSourceKey].Contains('multi-sample') then begin
        LStimuliFolder := Values[StimuliFolderKey];
        if LStimuliFolder.IsEmpty then begin
          LStimuliFolder := 'alpha_numeric_convention'
        end;
        Pool.ImageBasePath := AsPath(LStimuliFolder);
      end;
    end;
  end;
end;

initialization
  LastWord := EmptyWord;

end.

