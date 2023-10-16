{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.grids;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, fgl, SDL2, Math.LatinSquares, sdl.app.grids.types;

type

  { TGrid }
  TGrid = class
    private
      FFixedComparison: Boolean;
      FFixedSample: Boolean;
      FSeed : integer;
      FCellsCount: integer;
      FCellsSize: real;
      FComparisonsCount: integer;
      FGrid : TMatrix;
      FGridStyle : TGridStyle;
      FGridOrientation : TGridOrientation;
      FRandomPositions : TRandomPositions;
      FSamplesCount: integer;
      procedure SetCellsCount(AValue: integer);
      procedure SetCellsSize(AValue: real);
      procedure SetFixedComparison(AValue: Boolean);
      procedure SetFixedSample(AValue: Boolean);
      procedure SetGridOrientation(AValue: TGridOrientation);
      procedure SetGridStyle(AGridStyle: TGridStyle);
      procedure RandomizeGridList(AGridList: TGridList);
      function InvalidateGridList(IsSample: Boolean=true): TGridList;
      function DispersionStyle : Boolean;
      procedure CreatePositions;
    public
      constructor Create(ASeed : integer);
      destructor Destroy; override;
      function GetRandomGridOrientation : TGridOrientation;
      function IntToCell(AN : Cardinal) : TCell;
      function RectFromPosition(APosition: integer) : TRect;
      function PositionFromObject(AObject: TObject) : integer;
      function Header : string;
      function ToData : string;
      procedure UpdatePositions(ASamples, AComparisons: integer;
        AGridOrientation : TGridOrientation);
      {Cria seleção randômica de modelos e comparações em posições diferentes no AGrid}
      procedure RandomizePositions;
      property GridStyle : TGridStyle read FGridStyle write SetGridStyle;
      property CellsCount : integer read FCellsCount write SetCellsCount;
      property CellsSize : real read FCellsSize write SetCellsSize;
      property FixedSample : Boolean read FFixedSample write SetFixedSample;
      property FixedComparison : Boolean read FFixedComparison write SetFixedComparison;
      property RandomPositions : TRandomPositions read FRandomPositions;
      property Seed : integer read FSeed write FSeed;
      property Orientation: TGridOrientation read FGridOrientation write SetGridOrientation;
  end;

var
  Grid : TGrid = nil;

implementation

uses
  Math
  , sdl.app
  , sdl.app.grids.methods
  , sdl.app.stimulus.contract
  ;

{ TGrid }

procedure TGrid.SetGridStyle(AGridStyle: TGridStyle);
begin
  if FGridStyle = AGridStyle then Exit;
  FGridStyle := AGridStyle;
  case AGridStyle of
    gtCircle : FGrid := GetCircularCentralGrid(FSeed, FCellsSize);
    gtSquare : FGrid := GetCentralGrid(FSeed, FCellsSize, DispersionStyle);
    gtDistributed: FGrid := GetCentralGrid(FSeed, FCellsSize, DispersionStyle);
  end;
end;

procedure TGrid.RandomizeGridList(AGridList: TGridList);
var
  i : integer;
begin
  for i := AGridList.Count - 1 downto 0 do
    AGridList.Exchange(i, RandomRange(0, i + 1));
end;

function TGrid.InvalidateGridList(IsSample: Boolean): TGridList;
var
  i : integer;
begin
  Result:= TGridList.Create;
  {
    3x3
    0..1..2
    3..4..5
    6..7..8
  }
  case FGridOrientation of
      goNone: begin
        for i := 0 to FCellsCount - 1 do Result.Add(i);
      end;
      goLeftToRight: begin
        if IsSample then begin
          if FFixedSample then begin
            Result.Add(3);
          end else begin
            Result.Add(0);
            Result.Add(3);
            Result.Add(6);
          end;
        end else begin
          if FFixedComparison then begin
            Result.Add(5);
          end else begin
            Result.Add(2);
            Result.Add(5);
            Result.Add(8);
          end;
        end;
      end;
      goRightToLeft: begin
        if IsSample then begin
          if FFixedSample then begin
            Result.Add(5);
          end else begin
            Result.Add(2);
            Result.Add(5);
            Result.Add(8);
          end;
        end else begin
          if FFixedComparison then begin
            Result.Add(3);
          end else begin
            Result.Add(0);
            Result.Add(3);
            Result.Add(6);
          end;
        end;
      end;
      goBottomToTop: begin
        if IsSample then begin
          if FFixedSample then begin
            Result.Add(7);
          end else begin
            Result.Add(6);
            Result.Add(7);
            Result.Add(8);
          end;
        end else begin
          if FixedComparison then begin
            Result.Add(1);
          end else begin
            Result.Add(0);
            Result.Add(1);
            Result.Add(2);
          end;
        end;
      end;
      goTopToBottom: begin
        if IsSample then begin
          if FFixedSample then begin
            Result.Add(1);
          end else begin
            Result.Add(0);
            Result.Add(1);
            Result.Add(2);
          end;
        end else begin
          if FFixedComparison then begin
            Result.Add(7);
          end else begin
            Result.Add(6);
            Result.Add(7);
            Result.Add(8);
          end;
        end;
      end;
    end;
end;

function TGrid.DispersionStyle: Boolean;
begin
  case FGridStyle of
    gtCircle : Result := False; // it is ignored
    gtSquare : Result := False;
    gtDistributed : Result := True;
  end;
end;

procedure TGrid.CreatePositions;
var
  i : integer;
  LGridList : TGridList;
begin
  with FRandomPositions do begin
    SetLength(Samples, FSamplesCount);
    SetLength(Comparisons, FComparisonsCount);

    for i := low(Samples) to high(Samples) do
        Samples[i].Index := -1;
    for i := low(Comparisons) to high(Comparisons) do
        Comparisons[i].Index := -1;
    case FGridOrientation of
        goNone: begin
          // do nothing for now
        end;
        else begin
          LGridList := InvalidateGridList(True);
          SamplesRows := TLatinSquare.Create(FSeed);
          for i := 0 to LGridList.Count-1 do
            SamplesRows.Signs[i] := LGridList[i];
          SamplesRows.Invalidate;
          LGridList.Free;

          LGridList := InvalidateGridList(False);
          ComparisonsRows := TLatinSquare.Create(FSeed);
          for i := 0 to LGridList.Count-1 do
            ComparisonsRows.Signs[i] := LGridList[i];
          ComparisonsRows.Invalidate;
          LGridList.Free;
        end;
    end;
  end;
end;

procedure TGrid.RandomizePositions;
var
  Cell : TCell;
  LGridList : TGridList;

  {Change positions only}
  procedure SecureCopy(var A: TGridItem; B : TGridItem);
  begin
    A.Index := B.Index;
    A.Rect  := B.Rect;
    // A.Position := B.Position; // do not override position
    // A.Item := B.Item; // do not override Item Pointer
  end;

  procedure GridListToGridItems(var AGridItems : TGridItems);
  var
    i: Integer;
  begin
    for i := Low(AGridItems) to High(AGridItems) do
    begin
      AGridItems[i].Position := LGridList.First;
      Cell := IntToCell(AGridItems[i].Position);
      SecureCopy(AGridItems[i], FGrid[Cell[0], Cell[1]]);
      LGridList.Delete(0);
    end;
  end;

  procedure LatinRowToGridItems(ALatinSquare: TLatinSquare;
    var AGridItems : TGridItems);
  var
   LLatinRow : TLatinRow;
   i : integer;
  begin
    LLatinRow := ALatinSquare.NextRow;
    LGridList := TGridList.Create;
    for i in LLatinRow do LGridList.Add(i);
    //RandomizeGridList(LGridList);
    GridListToGridItems(AGridItems);
    LGridList.Free;
  end;

begin
  with FRandomPositions do begin
    case FGridOrientation of
      goNone: begin
        LGridList:= InvalidateGridList;
        RandomizeGridList(LGridList);
        GridListToGridItems(Samples);
        GridListToGridItems(Comparisons);
        LGridList.Free;
      end;
      else begin
        if FFixedSample then begin
          LGridList := InvalidateGridList(True);
          Samples[Low(Samples)].Position := LGridList.First;
          Cell := IntToCell(Samples[Low(Samples)].Position);
          SecureCopy(Samples[Low(Samples)], FGrid[Cell[0], Cell[1]]);
        end else begin
          LatinRowToGridItems(SamplesRows, Samples);
        end;
        if FFixedComparison then begin
          LGridList := InvalidateGridList(False);
          Comparisons[Low(Comparisons)].Position := LGridList.First;
          Cell := IntToCell(Comparisons[Low(Comparisons)].Position);
          SecureCopy(Comparisons[Low(Comparisons)], FGrid[Cell[0], Cell[1]]);
        end else begin
          LatinRowToGridItems(ComparisonsRows, Comparisons);
        end;
      end;
    end;
  end;
end;

function TGrid.RectFromPosition(APosition: integer): TRect;
var
  j, i: Integer;
begin
  for j := Low(FGrid) to High(FGrid) do begin
    for i := Low(FGrid[j]) to High(FGrid[j]) do begin
      with FGrid[j][i] do begin
        if Index = APosition then begin
          Result := Classes.Rect(Rect.x, Rect.y, Rect.x+Rect.w, Rect.y+Rect.h);
        end;
      end;
    end;
  end;
end;

function TGrid.PositionFromObject(AObject: TObject): integer;
var
  i: Integer;
  Cell: TCell;
begin
  Result := -1;
  for i := 0 to 8 do
  begin
    Cell := IntToCell(i);
    if FGrid[Cell[0], Cell[1]].Item = AObject then begin
      Result:= i;
      Exit;
    end;
  end;
end;

function TGrid.Header: string;
var
  i: Integer;
begin
  with FRandomPositions do begin
    Result := '';
    for i := Low(Samples) to High(Samples) do begin
      if i = 0 then begin
        Result := String.Join('', [Result, 'Sample-Position.'+(i+1).ToString]);
      end else begin
        Result := String.Join(#9, [Result, 'Sample-Position.'+(i+1).ToString]);
      end;
    end;

    for i := Low(Comparisons) to High(Comparisons) do begin
        Result := String.Join(#9, [Result, 'Comparison-Position.'+(i+1).ToString]);
    end;
  end;
end;

function TGrid.ToData: string;
var
  i: Integer;
  LIStimulus : IStimulus;
begin
  Result := '';
  with FRandomPositions do begin
    for i := Low(Samples) to High(Samples) do begin
      LIStimulus := Samples[i].Item as IStimulus;
      if i = 0 then begin
        Result := String.Join('', [Result, LIStimulus.ToData]);
      end else begin
        Result := String.Join(#9, [Result, LIStimulus.ToData]);
      end;
    end;

    for i := Low(Comparisons) to High(Comparisons) do begin
      LIStimulus := Comparisons[i].Item as IStimulus;
      Result := String.Join(#9, [Result, LIStimulus.ToData]);
    end;
  end;
end;

procedure TGrid.SetCellsCount(AValue: integer);
begin
  if FCellsCount=AValue then Exit;
  FCellsCount:=AValue;
end;

procedure TGrid.SetCellsSize(AValue: real);
begin
  if FCellsSize=AValue then Exit;
  FCellsSize:=AValue;
end;

procedure TGrid.SetFixedComparison(AValue: Boolean);
begin
  if FFixedComparison=AValue then Exit;
  FFixedComparison:=AValue;
end;

procedure TGrid.SetFixedSample(AValue: Boolean);
begin
  if FFixedSample=AValue then Exit;
  FFixedSample:=AValue;
end;

procedure TGrid.SetGridOrientation(AValue: TGridOrientation);
begin
  if FGridOrientation=AValue then Exit;
  FGridOrientation:=AValue;
end;

function TGrid.GetRandomGridOrientation : TGridOrientation;
var
  i: integer;
begin
  i:= RandomRange(1, 5);
  Result := TGridOrientation(i);
end;

function TGrid.IntToCell(AN: Cardinal): TCell;
const
  Col = 0;
  Row = 1;
begin
  if AN > ((FSeed * FSeed) -1) then begin
    raise Exception.Create('Unknown position');
  end;

  Result[Col] := AN div FSeed;  // Row
  Result[Row] := AN mod FSeed;  // Column
end;

constructor TGrid.Create(ASeed: integer);
begin
  InitMonitor;
  FSeed := ASeed;
  FSamplesCount := -1;
  FComparisonsCount := -1;
  FCellsCount:=ASeed*ASeed;
  FCellsSize := 6;
  FFixedSample := True;
  FFixedComparison:=False;
  FGridStyle := gtDistributed;
  FGridOrientation:= goTopToBottom;
  FGrid := GetCentralGrid(FSeed, FCellsSize, DispersionStyle);
end;

destructor TGrid.Destroy;
begin
  with FRandomPositions do
  begin
    SamplesRows.Free;
    ComparisonsRows.Free;
  end;
end;

procedure TGrid.UpdatePositions(ASamples, AComparisons : integer;
  AGridOrientation : TGridOrientation);
begin
  if (FSamplesCount <> ASamples) or
     (FComparisonsCount <> AComparisons) or
     (FGridOrientation <> AGridOrientation) then begin
    FSamplesCount := ASamples;
    FComparisonsCount := AComparisons;
    FGridOrientation := AGridOrientation;
    CreatePositions;
  end;
  RandomizePositions;
end;

end.

