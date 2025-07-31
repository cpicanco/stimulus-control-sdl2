{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.trials.reinforcement;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

type
  TBlockReinforcement = array of Boolean;

function BlockReinforcement(ATotalTrials: integer;
   AReinforcementPorcentage : integer) : TBlockReinforcement;

implementation

uses session.shuffler.types, session.shuffler.base;

function BlockReinforcement(ATotalTrials: integer;
  AReinforcementPorcentage: integer): TBlockReinforcement;
var
  i : integer;
  LTotalReinforcement : integer;
  LReferenceList : TReferenceList;
  LSortedList : TBlockReinforcement;
  LShuffler : TShuffler;
  LItem : TItem;
begin
  Result := Default(TBlockReinforcement);
  LTotalReinforcement := (ATotalTrials * AReinforcementPorcentage) div 100;
  SetLength(LSortedList, ATotalTrials);
  SetLength(Result, ATotalTrials);
  for i := 0 to LTotalReinforcement - 1 do
    LSortedList[i] := True;
  for i := LTotalReinforcement to ATotalTrials - 1 do
    LSortedList[i] := False;

  LReferenceList := TReferenceList.Create;
  LShuffler := TShuffler.Create;
  try
    for i := Low(LSortedList) to High(LSortedList) do begin
      LItem.ID := i;
      LItem.ReferenceName := LSortedList[i].ToString;
      LReferenceList.Add(LItem);
    end;
    LShuffler.Shuffle(LReferenceList);
    for i := Low(LSortedList) to High(LSortedList) do begin
      Result[i] := LSortedList[LShuffler.Values(i)];
    end;
  finally
    LReferenceList.Free;
    LShuffler.Free;
  end;
end;



end.

