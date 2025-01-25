{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.shuffler.base;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, Generics.Collections, session.shuffler.types;

type

  TIntegerList = specialize TList<integer>;

  { TShuffler }

  TShuffler = class
  private
    FShuffledList: TIntegerList;
    procedure SimpleShuffle(AList: TReferenceList);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Shuffle(AList: TReferenceList);
    function Values(AIndex: Integer): Integer;
    function ToArray : TIntArray;
  end;

implementation

uses session.pool;

type

  TUniqueCount = specialize TDictionary<string, integer>;


constructor TShuffler.Create;
begin
  FShuffledList := TIntegerList.Create;
end;

destructor TShuffler.Destroy;
begin
  FShuffledList.Free;
  inherited Destroy;
end;

procedure TShuffler.SimpleShuffle(AList: TReferenceList);
var
  i, j : Integer;
begin
  FShuffledList.Clear;
  if AList.Count = 1 then begin
    FShuffledList.Add(0);
    Exit;
  end;

  for i := 0 to AList.Count - 1 do begin
    FShuffledList.Add(i);
  end;

  for i := 0 to FShuffledList.Count - 1 do begin
    j := Random(FShuffledList.Count);
    FShuffledList.Exchange(i, j);
  end;
end;

function TShuffler.Values(AIndex: Integer): Integer;
begin
  if (AIndex >= 0) and (AIndex < FShuffledList.Count) then
    Result := FShuffledList[AIndex]
  else begin
    raise EArgumentOutOfRangeException.Create(
      'Block '+(Pool.Block.ID+1).ToString+
      ' TShuffler.Value: '+AIndex.ToString+
      ' FShuffledList.Count: '+FShuffledList.Count.ToString);
  end;
end;

function TShuffler.ToArray: TIntArray;
var
  i: Integer;
begin
  Result := nil;
  SetLength(Result, FShuffledList.Count);
  for i := 0 to FShuffledList.Count-1 do begin
    Result[i] := FShuffledList[i];
  end;
end;

procedure TShuffler.Shuffle(AList: TReferenceList);
var
  LUniqueCount: TUniqueCount;
  LTempList : TReferenceList;
  LReferenceList : TReferenceList;
  i : integer;

  function RandomUniqueNames(AReferenceList : TReferenceList) : Boolean;
  var
    i : integer;
    Item : TItem;

    function IndexFromName(AName : string; out AID : integer) : Boolean;
    var
      i : integer;
    begin
      Result := False;
      for i := 0 to AList.Count-1 do begin
        if AList[i].ReferenceName = AName then begin
          AID := AList[i].ID;
          AList.Delete(i);
          Result := True;
          Break;
        end;
      end;
    end;
  begin
    Result := False;
    if LUniqueCount.Count = 0 then Exit;
    AReferenceList.Clear;
    for i := LUniqueCount.Count-1 downto 0 do begin
      Item.ReferenceName := LUniqueCount.Keys.ToArray[i];
      IndexFromName(Item.ReferenceName, Item.ID);
      AReferenceList.Add(Item);
      LUniqueCount[Item.ReferenceName] := LUniqueCount[Item.ReferenceName] -1;
      if LUniqueCount[Item.ReferenceName] = 0 then begin
        LUniqueCount.Remove(Item.ReferenceName);
      end;
    end;
    for i := 0 to AReferenceList.Count-1 do begin
      AReferenceList.Exchange(i, Random(AReferenceList.Count));
    end;
    Result := LUniqueCount.Count > 0;
  end;
begin
  LTempList := TReferenceList.Create;
  LReferenceList := TReferenceList.Create;
  LUniqueCount := TUniqueCount.Create;
  try
    // Count the occurrences of each unique string
    for i := 0 to AList.Count-1 do begin
      if LUniqueCount.ContainsKey(AList[i].ReferenceName) then begin
        LUniqueCount[AList[i].ReferenceName] :=
          LUniqueCount[AList[i].ReferenceName] + 1;
      end else begin
        LUniqueCount.Add(AList[i].ReferenceName, 1);
      end;
    end;

    if (LUniqueCount.Count = 1) or
       (LUniqueCount.Count = AList.Count) or
       ((LUniqueCount.Count = 2) and (AList.Count <= 4)) then begin
      SimpleShuffle(AList);
      Exit;
    end;

    // if LUniqueCount.Count > AList.Count then begin // Should never occur

    repeat
      RandomUniqueNames(LTempList);
      for i := 0 to LTempList.Count -1 do begin
        LReferenceList.Add(LTempList[i]);
      end;
    until LUniqueCount.Count = 0;

    FShuffledList.Clear;
    for i := 0 to LReferenceList.Count -1 do begin
      FShuffledList.Add(LReferenceList[i].ID);
    end;
  finally
    LTempList.Free;
    LReferenceList.Free;
    LUniqueCount.Free;
  end;
end;

end.

