{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.grids;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2, Math, Math.LatinSquares, sdl.app.grids.types;

type

  { TGrid }
  TGrid = class
    private
      FFixedComparison: Boolean;
      FFixedSample: Boolean;
      //FSampleGridList : TGridList;
      //FComparGridList : TGridList;
      FSeed : integer;
      FCellsCount: integer;
      FCellsSizeInCentimenter: Float;
      FComparisonsCount: integer;
      FGrid : TMatrix;
      FGridStyle : TGridStyle;
      FGridOrientation : TGridOrientation;
      FRandomPositions : TRandomPositions;
      FSamplesCount: integer;
      procedure SetCellsCount(AValue: integer);
      procedure SetCellsSize(AValue: Float);
      procedure SetFixedComparison(AValue: Boolean);
      procedure SetFixedSample(AValue: Boolean);
      procedure SetGridOrientation(AValue: TGridOrientation);
      procedure SetGridStyle(AGridStyle: TGridStyle);
      procedure RandomizeGridList(AGridList: TGridList);
      procedure InvalidateSampleGridList(AGridList: TGridList);
      procedure InvalidateComparGridList(AGridList: TGridList);
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
      function ToJSON : string;
      procedure UpdatePositions(AGridSize: Byte;
        ASamples, AComparisons: integer;
        AGridOrientation : TGridOrientation;
        AFixedSample: Boolean; AFixedComparison: Boolean);
      {Cria seleção randômica de modelos e comparações em posições diferentes no AGrid}
      procedure RandomizePositions;
      property GridStyle : TGridStyle read FGridStyle write SetGridStyle;
      property CellsCount : integer read FCellsCount write SetCellsCount;
      property CellsSize : Float read FCellsSizeInCentimenter write SetCellsSize;
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
  sdl.helpers,
  session.parameters.global,
  sdl.app.grids.methods,
  sdl.app.stimulus.contract;

{ TGrid }

procedure TGrid.SetGridStyle(AGridStyle: TGridStyle);
begin
  if FGridStyle = AGridStyle then Exit;
  FGridStyle := AGridStyle;
  case AGridStyle of
    gtCircle : FGrid := GetCircularCentralGrid(FSeed, FCellsSizeInCentimenter);
    gtSquare : FGrid := GetCentralGrid(FSeed, FCellsSizeInCentimenter, DispersionStyle);
    gtDistributed: FGrid := GetCentralGrid(FSeed, FCellsSizeInCentimenter, DispersionStyle);
  end;
end;

procedure TGrid.RandomizeGridList(AGridList: TGridList);
var
  i : integer;
begin
  for i := AGridList.Count - 1 downto 0 do
    AGridList.Exchange(i, RandomRange(0, i + 1));
end;

procedure TGrid.InvalidateSampleGridList(AGridList: TGridList);
var
  i: Integer;
  LOrientation : string;
begin
  WriteStr(LOrientation, FGridOrientation);
  LOrientation :=
    'Not implemented with '+ LOrientation + ' orientation and ' +
    'fixed sample = ' + FFixedSample.ToString;
  AGridList.Clear;

  if FFixedSample then begin
    case FGridOrientation of
      goCustom :
        AGridList.Add(GlobalTrialParameters.FixedSamplePosition);
      else begin
        raise ENotImplemented.Create(LOrientation)
      end;
    end;
  end else begin
    case FGridOrientation of
      goNone: begin
        for i := 0 to FCellsCount - 1 do AGridList.Add(i);
      end;

      goLeftToRight: begin
        case FSeed of
          3 : begin
            AGridList.Add(0);
            AGridList.Add(3);
            AGridList.Add(6);
          end;

          4 : begin
            AGridList.Add(0);
            AGridList.Add(5);
            AGridList.Add(10);
            AGridList.Add(15);
            AGridList.Add(20);
          end;
        end;
      end;

      goRightToLeft: begin
        case FSeed of
          3 : begin
            AGridList.Add(2);
            AGridList.Add(5);
            AGridList.Add(8);
          end;

          4 : begin
            AGridList.Add(4);
            AGridList.Add(9);
            AGridList.Add(14);
            AGridList.Add(19);
            AGridList.Add(24);
          end;
        end;
      end;

      goBottomToTop: begin
        case FSeed of
          3 : begin
            AGridList.Add(6);
            AGridList.Add(7);
            AGridList.Add(8);
          end;

          4 : begin
            AGridList.Add(20);
            AGridList.Add(21);
            AGridList.Add(22);
            AGridList.Add(23);
            AGridList.Add(24);
          end;
        end;
      end;

      goTopToBottom: begin
        case FSeed of
          3 : begin
            AGridList.Add(0);
            AGridList.Add(1);
            AGridList.Add(2);
          end;

          4 : begin
            AGridList.Add(0);
            AGridList.Add(1);
            AGridList.Add(2);
            AGridList.Add(3);
            AGridList.Add(4);
          end;
        end;
      end;

      goCustom : begin
        raise ENotImplemented.Create(LOrientation)
      end;
    end;
  end;
end;

procedure TGrid.InvalidateComparGridList(AGridList: TGridList);
var
  i: Integer;
begin
  AGridList.Clear;
  if FFixedComparison then begin
    case FGridOrientation of
      goCustom :
        AGridList.Add(GlobalTrialParameters.FixedComparisonPosition);
      else begin
        raise ENotImplemented.Create(
          'Fixed comparison not implemented with this orientation.')
      end;
    end;
  end else begin
    case FGridOrientation of
      goNone: begin
        for i := 0 to FCellsCount - 1 do AGridList.Add(i);
      end;
      goLeftToRight: begin
        case FSeed of
          3 : begin
            AGridList.Add(2);
            AGridList.Add(5);
            AGridList.Add(8);
          end;

          4 : begin
            AGridList.Add(4);
            AGridList.Add(9);
            AGridList.Add(14);
            AGridList.Add(19);
            AGridList.Add(24);
          end;
        end;


      end;
      goRightToLeft: begin
        case FSeed of
          3 : begin
            AGridList.Add(0);
            AGridList.Add(3);
            AGridList.Add(6);
          end;

          4 : begin
            AGridList.Add(0);
            AGridList.Add(5);
            AGridList.Add(10);
            AGridList.Add(15);
            AGridList.Add(20);
          end;
        end;


      end;
      goBottomToTop: begin
        case FSeed of
          3 : begin
            AGridList.Add(0);
            AGridList.Add(1);
            AGridList.Add(2);
          end;

          4 : begin
            AGridList.Add(0);
            AGridList.Add(1);
            AGridList.Add(2);
            AGridList.Add(3);
            AGridList.Add(4);
          end;
        end;


      end;
      goTopToBottom: begin
        case FSeed of
          3 : begin
            AGridList.Add(6);
            AGridList.Add(7);
            AGridList.Add(8);
          end;

          4 : begin
            AGridList.Add(20);
            AGridList.Add(21);
            AGridList.Add(22);
            AGridList.Add(23);
            AGridList.Add(24);
          end;
        end;

      end;
      goCustom: begin
        with GlobalTrialParameters do
          for i := Low(ComparisonPositions) to High(ComparisonPositions) do begin
            AGridList.Add(ComparisonPositions[i]);
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
          { do nothing }
        end;

        goCustom: begin
          LGridList := TGridList.Create;
          try
            //InvalidateSampleGridList(LGridList);
            //if Assigned(SamplesRows) then begin
            //  SamplesRows.Free;
            //end;
            //SamplesRows := TLatinSquare.Create(FSeed);
            //for i := 0 to LGridList.Count-1 do
            //  SamplesRows.Signs[i] := LGridList[i];
            //SamplesRows.Invalidate;


            InvalidateComparGridList(LGridList);

            if Assigned(ComparisonsRows) then begin
              ComparisonsRows.Free;
            end;
            ComparisonsRows := TLatinSquare.Create(LGridList.Count);
            for i := 0 to LGridList.Count-1 do
              ComparisonsRows.Signs[i] := LGridList[i];
            ComparisonsRows.Invalidate;
          finally
            LGridList.Free;
          end;
        end;

        else begin
          LGridList := TGridList.Create;
          try
            InvalidateSampleGridList(LGridList);
            if Assigned(SamplesRows) then begin
              SamplesRows.Free;
            end;
            SamplesRows := TLatinSquare.Create(FSeed);
            for i := 0 to LGridList.Count-1 do
              SamplesRows.Signs[i] := LGridList[i];
            SamplesRows.Invalidate;


            InvalidateComparGridList(LGridList);

            if Assigned(ComparisonsRows) then begin
              ComparisonsRows.Free;
            end;
            ComparisonsRows := TLatinSquare.Create(FSeed);
            for i := 0 to LGridList.Count-1 do
              ComparisonsRows.Signs[i] := LGridList[i];
            ComparisonsRows.Invalidate;
          finally
            LGridList.Free;
          end;
        end;
    end;
  end;
end;

procedure TGrid.RandomizePositions;
var
  Cell : TCell;
  LGridList : TGridList;
  n : integer;

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
    LGridList.Clear;
    for i in LLatinRow do LGridList.Add(i);
    //RandomizeGridList(LGridList);
    GridListToGridItems(AGridItems);
  end;

begin
  with FRandomPositions do begin
    LGridList := TGridList.Create;
    try
      case FGridOrientation of
        goNone: begin
          LGridList.Clear;
          for n := 0 to FCellsCount - 1 do LGridList.Add(n);
          RandomizeGridList(LGridList);
          GridListToGridItems(Samples);
          GridListToGridItems(Comparisons);
        end;

        goCustom: begin
          if FFixedSample then begin
            Samples[Low(Samples)].Position :=
              GlobalTrialParameters.FixedSamplePosition;
            Cell := IntToCell(Samples[Low(Samples)].Position);
            SecureCopy(Samples[Low(Samples)], FGrid[Cell[0], Cell[1]]);
          end else begin
            raise ENotImplemented.Create(
              'Random sample positions with custom grid orientation.');
          end;

          if FFixedComparison then begin
            Comparisons[Low(Comparisons)].Position :=
              GlobalTrialParameters.FixedComparisonPosition;
            Cell := IntToCell(Comparisons[Low(Comparisons)].Position);
            SecureCopy(Comparisons[Low(Comparisons)], FGrid[Cell[0], Cell[1]]);
          end else begin
            LatinRowToGridItems(ComparisonsRows, Comparisons);
          end;
        end;

        else begin
          if FFixedSample then begin
            Samples[Low(Samples)].Position :=
              GlobalTrialParameters.FixedSamplePosition;
            Cell := IntToCell(Samples[Low(Samples)].Position);
            SecureCopy(Samples[Low(Samples)], FGrid[Cell[0], Cell[1]]);
          end else begin
            LatinRowToGridItems(SamplesRows, Samples);
          end;

          if FFixedComparison then begin
            //InvalidateComparGridList(LGridList);
            //Comparisons[Low(Comparisons)].Position := LGridList.First;
            //Cell := IntToCell(Comparisons[Low(Comparisons)].Position);
            Comparisons[Low(Comparisons)].Position :=
              GlobalTrialParameters.FixedComparisonPosition;
            Cell := IntToCell(Comparisons[Low(Comparisons)].Position);
            SecureCopy(Comparisons[Low(Comparisons)], FGrid[Cell[0], Cell[1]]);
          end else begin
            LatinRowToGridItems(ComparisonsRows, Comparisons);
          end;
        end;
      end;
    finally
      LGridList.Free;
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

function TGrid.ToJSON: string;
var
  j, i: Integer;
  LSeparator : string;
begin
  Result := '';
  for j := Low(FGrid) to High(FGrid) do begin
    for i := Low(FGrid[j]) to High(FGrid[j]) do begin
      with FGrid[j][i] do begin
        if (j = 0) and
           (i = 0) then begin
          LSeparator := '';
        end else begin
          LSeparator := ',';
        end;
        Result := String.Join(LSeparator, [Result,
          Index.ToString + ':' + Rect.ToJSON])
      end;
    end;
  end;
  Result := '{grid:{'+Result+'}}'
end;

procedure TGrid.SetCellsCount(AValue: integer);
begin
  if FCellsCount=AValue then Exit;
  FCellsCount:=AValue;
end;

procedure TGrid.SetCellsSize(AValue: Float);
begin
  if FCellsSizeInCentimenter=AValue then Exit;
  FCellsSizeInCentimenter:=AValue;
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

  Result[Col] := AN div FSeed;
  Result[Row] := AN mod FSeed;
end;

constructor TGrid.Create(ASeed: integer);
begin
  InitMonitor;
  FSeed := ASeed;
  FSamplesCount := -1;
  FComparisonsCount := -1;
  FCellsCount:=ASeed*ASeed;
  FCellsSizeInCentimenter := GlobalTrialParameters.CellsSizeInCentimenter;
  FFixedSample := True;
  FFixedComparison:=False;
  FGridStyle := gtDistributed;
  FGridOrientation:= goTopToBottom;
  FGrid := GetCentralGrid(FSeed, FCellsSizeInCentimenter, DispersionStyle);
end;

destructor TGrid.Destroy;
begin
  with FRandomPositions do
  begin
    SamplesRows.Free;
    ComparisonsRows.Free;
  end;
end;

procedure TGrid.UpdatePositions(AGridSize: Byte;
  ASamples, AComparisons : integer;
  AGridOrientation : TGridOrientation;
  AFixedSample: Boolean; AFixedComparison: Boolean);
begin
  if (FSeed <> AGridSize) or
     (FSamplesCount <> ASamples) or
     (FComparisonsCount <> AComparisons) or
     (FGridOrientation <> AGridOrientation) or
     (FFixedSample <> AFixedSample) or
     (FFixedComparison <> AFixedComparison) then begin
    FFixedSample := AFixedSample;
    FFixedComparison := AFixedComparison;
    FSamplesCount := ASamples;
    FComparisonsCount := AComparisons;
    FGridOrientation := AGridOrientation;

    if FSeed <> AGridSize then begin
      FSeed := AGridSize;
      FCellsCount:=FSeed*FSeed;
      FGrid := GetCentralGrid(FSeed, FCellsSizeInCentimenter, DispersionStyle);
    end;
    CreatePositions;
  end;
  RandomizePositions;
end;

end.

