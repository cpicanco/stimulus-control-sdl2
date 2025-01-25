{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit math.latinsquare;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

type
  TLatinSquare = array of array of Integer;
  TIntArray = array of integer;

procedure NewLatinSquare(n: Integer; var ALatinSquare: TLatinSquare);
function IsLatinSquare(const latinSquare: TLatinSquare): Boolean;
procedure PrintLatinSquare(const latinSquare: TLatinSquare);

implementation

uses Math;

function ToArray(const intSet: TIntegerSet): TIntArray;
var
  i, count: Integer;
begin
  count := 0;
  SetLength(Result, Length(intSet));
  for i := 0 to High(Result) do
  begin
    if i in intSet then
    begin
      Result[count] := i;
      Inc(count);
    end;
  end;
  SetLength(Result, count);
end;

procedure RandomizeArray(var arr: array of Integer);
var
  i, j, temp: Integer;
begin
  //Randomize;
  for i := High(arr) downto 1 do
  begin
    j := Random(i + 1);
    temp := arr[j];
    arr[j] := arr[i];
    arr[i] := temp;
  end;
end;

procedure NewLatinSquare(n: Integer; var ALatinSquare: TLatinSquare);
var
  j : Integer;
  AvailableSymbols: array of set of 0..255;

  function IsSafe(const r, c, num: Integer): Boolean;
  var
    k: Integer;
  begin
    // Check if the symbol num is already present in the current row or column
    for k := 0 to n - 1 do
    begin
      if ALatinSquare[r, k] = num then
        Exit(False);
      if ALatinSquare[k, c] = num then
        Exit(False);
    end;
    Result := True;
  end;

  function Backtrack(const r, c: Integer): Boolean;
  var
    avail : set of 0..255;
    i, num: Integer;
  begin
    if r = n then
      Exit(True); // All cells have been filled, a valid Latin square is found

    if c = n then
      Exit(Backtrack(r + 1, 0)); // Move to the next row

    if ALatinSquare[r, c] <> 0 then
      Exit(Backtrack(r, c + 1)); // Cell is already filled, move to the next cell

    avail := AvailableSymbols[c];
    for i := 0 to n-1 do
    begin
      num := avail random(n);
      if IsSafe(r, c, num) then
      begin
        ALatinSquare[r, c] := num; // Assign the symbol num to the cell

        if Backtrack(r, c + 1) then
          Exit(True); // Continue filling the next cell

        ALatinSquare[r, c] := 0; // Undo the assignment if it does not lead to a valid solution
      end;
    end;

    Result := False; // Backtrack to the previous cell and try a different symbol
  end;

begin
  SetLength(ALatinSquare, n, n);
  SetLength(AvailableSymbols, n);

  for j := 0 to n - 1 do
    AvailableSymbols[j] := [1..n];

  if Backtrack(0, 0) then
    Writeln('A valid Latin square is found.')
  else
    Writeln('Failed to generate a valid Latin square.');
end;


function IsLatinSquare(const latinSquare: TLatinSquare): Boolean;
var
  n, i, j, x: Integer;
  rowSet, colSet: set of 0..255;
begin
  n := Length(latinSquare);

  // Check rows
  for i := 0 to n - 1 do
  begin
    rowSet := [];
    for j := 0 to n - 1 do
    begin
      x := latinSquare[i, j];
      if x in rowSet then
        Exit(False);
      Include(rowSet, x);
    end;
  end;

  // Check columns
  for j := 0 to n - 1 do
  begin
    colSet := [];
    for i := 0 to n - 1 do
    begin
      x := latinSquare[i, j];
      if x in colSet then
        Exit(False);
      Include(colSet, x);
    end;
  end;

  // All rows and columns have distinct symbols
  Result := True;
end;

procedure PrintLatinSquare(const latinSquare: TLatinSquare);
var
  n, i, j: Integer;
begin
  n := Length(latinSquare);

  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
      Write(latinSquare[i, j], ' ');
    Writeln;
  end;
end;

end.
