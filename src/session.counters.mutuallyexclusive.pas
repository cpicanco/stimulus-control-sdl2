{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters.mutuallyexclusive;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, fgl, session.counters.consecutive;

type

  TCounterType = (CounterHit, CounterMiss, CounterNone);

  TCounters = specialize TFPGList<TConsecutivesCounter>;

  { TMutuallyExclusiveCounters }

  TMutuallyExclusiveCounters = class
  private
    FLastCounter : TCounterType;
    FCounters : TCounters;
    procedure Update(ACounterType : TCounterType);
    procedure Flush;
  public
    class function Header : string;
    constructor Create;
    destructor Destroy; override;
    procedure Reset;
    procedure Invalidate;
    procedure Hit;
    procedure Miss;
    procedure None;
    function Last : string;
    function Hits : TConsecutivesCounter;
    function Misses : TConsecutivesCounter;
    function Nones : TConsecutivesCounter;
    function ToIni : string;
    function ToData: string;
  end;

implementation

uses session.strutils;

{ TCounters }

procedure TMutuallyExclusiveCounters.Update(ACounterType: TCounterType);
var
  LCounter : TConsecutivesCounter;
  L : TConsecutivesCounter;
begin
  FLastCounter := ACounterType;
  LCounter := FCounters[Ord(ACounterType)];
  LCounter.Next;
  LCounter.NextConsecutive;
  for L in FCounters do begin
    if L = LCounter then
      Continue;
    L.ResetConsecutive;
  end;
end;

procedure TMutuallyExclusiveCounters.Flush;
var
  LCounter : TConsecutivesCounter;
begin
  for LCounter in FCounters do begin
    LCounter.FlushMaxConsecutives;
  end;
end;

constructor TMutuallyExclusiveCounters.Create;
var
  i: TCounterType;
begin
  FCounters := TCounters.Create;
  for i := Low(TCounterType) to High(TCounterType) do begin
    FCounters.Add(TConsecutivesCounter.Create);
  end;
end;

destructor TMutuallyExclusiveCounters.Destroy;
var
  LCounter : TConsecutivesCounter;
begin
  if FCounters.Count > 0 then begin
    for LCounter in FCounters do begin
      LCounter.Free;
    end;
  end;
  FCounters.Free;
  inherited Destroy;
end;

procedure TMutuallyExclusiveCounters.Hit;
begin
  Update(CounterHit);
end;

procedure TMutuallyExclusiveCounters.Miss;
begin
  Update(CounterMiss);
end;

procedure TMutuallyExclusiveCounters.None;
begin
  Update(CounterNone);
end;

function TMutuallyExclusiveCounters.Last: string;
begin
  WriteStr(Result, FLastCounter);
  Result := Result.Replace('Counter', '');
end;

class function TMutuallyExclusiveCounters.Header: string;
var
  LCounterType : TCounterType;
  LCounterTypeString : string;
begin
  Result := 'Result';
  for LCounterType in [Low(TCounterType)..High(TCounterType)] do begin
    WriteStr(LCounterTypeString, LCounterType);
    Result := String.Join(#9, [Result,
      LCounterTypeString, LCounterTypeString+'.MaxConsecutives']);
  end;
end;

procedure TMutuallyExclusiveCounters.Reset;
var
  LCounter : TConsecutivesCounter;
begin
  for LCounter in FCounters do begin
    LCounter.ResetConsecutive;
  end;
end;

procedure TMutuallyExclusiveCounters.Invalidate;
var
  LCounter : TConsecutivesCounter;
begin
  for LCounter in FCounters do begin
    LCounter.Invalidate;
  end;
end;

function TMutuallyExclusiveCounters.Hits: TConsecutivesCounter;
begin
  Result := FCounters[Ord(CounterHit)];
end;

function TMutuallyExclusiveCounters.Misses: TConsecutivesCounter;
begin
  Result := FCounters[Ord(CounterMiss)];
end;

function TMutuallyExclusiveCounters.Nones: TConsecutivesCounter;
begin
  Result := FCounters[Ord(CounterNone)];
end;

function TMutuallyExclusiveCounters.ToIni: string;
var
  LCounterType : TCounterType;
  LCounterTypeString : string;
  LCounter : TConsecutivesCounter;
begin
  Result := '';
  for LCounterType in [Low(TCounterType)..High(TCounterType)] do begin
    WriteStr(LCounterTypeString, LCounterType);
    LCounter := FCounters[Ord(LCounterType)];
    Result := Result +
      KeyValue(LCounterTypeString, LCounter.Count.ToString) +
      KeyValue(LCounterTypeString+'.MaxConsecutives',
        LCounter.MaxConsecutives.ToString);
  end;
end;

function TMutuallyExclusiveCounters.ToData: string;
var
  LCounterType : TCounterType;
  LCounterTypeString : string;
  LCounter : TConsecutivesCounter;
begin
  Self.Flush;
  Result := '';
  for LCounterType in [Low(TCounterType)..High(TCounterType)] do begin
    WriteStr(LCounterTypeString, LCounterType);
    LCounter := FCounters[Ord(LCounterType)];
    if LCounterType = Low(TCounterType) then begin
      Result := String.Join(#9, [
        LCounter.Count.ToString, LCounter.MaxConsecutives.ToString]);
    end else begin
      Result := String.Join(#9, [Result,
        LCounter.Count.ToString, LCounter.MaxConsecutives.ToString]);
    end;
  end;
end;

end.

