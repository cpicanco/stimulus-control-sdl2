{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Generics.Iterator.Table;

interface

uses SysUtils
   , Generics.Iterator.Table.Contract
   , Generics.Aggregator.Table.Contract
   , Generics.Tables.Types;

type

  { TTableIterator }

  generic TTableIterator<_GT> = class(specialize ITableIterator<_GT>)
  public type
    IAggregatorSpec = specialize ITableAggregator<_GT>;
  private
    FTableAggregator : IAggregatorSpec;
    FSavedRow : UInt32;
    FSavedCol : UInt32;
    FRowIndex : UInt32;
    FColIndex : UInt32;
    function HighCol : UInt32;
    function HighRow : UInt32;
  public
    constructor Create(AAggregator: IAggregatorSpec);
    function GetCurrent : _GT;
    function IsCurrentEmpty(out ACell : _GT) : Boolean; overload;
    function IsFirstRow: Boolean;
    function IsLastRow: Boolean;
    function IsFirstCol: Boolean;
    function IsLastCol: Boolean;
    procedure GoToCell(ACell : TCell);
    procedure GoFirstRow;
    procedure GoNextRow;
    procedure GoPreviousRow;
    procedure GoLastRow;
    procedure GoFirstCol;
    procedure GoNextCol;
    procedure GoPreviousCol;
    procedure GoLastCol;
    procedure Save;
    procedure Load;
  end;

implementation

{ TTableIterator }

function TTableIterator.HighCol: UInt32;
begin
  Result := FTableAggregator.Table.ColCount-1;
end;

function TTableIterator.HighRow: UInt32;
begin
  Result := FTableAggregator.Table.RowCount-1;
end;

constructor TTableIterator.Create(AAggregator: IAggregatorSpec);
begin
  FTableAggregator := AAggregator;
end;

function TTableIterator.GetCurrent: _GT;
begin
  Result := FTableAggregator.Table.Cells[FColIndex, FRowIndex];
end;

function TTableIterator.IsCurrentEmpty(out ACell : _GT): Boolean;
begin
  Result := FTableAggregator.Table.IsCellEmpty(FColIndex, FRowIndex, ACell);
end;

function TTableIterator.IsFirstRow: Boolean;
begin
  Result := FRowIndex = 0;
end;

function TTableIterator.IsLastRow: Boolean;
begin
  Result := FRowIndex = HighRow;
end;

function TTableIterator.IsFirstCol: Boolean;
begin
  Result := FColIndex = 0;
end;

function TTableIterator.IsLastCol: Boolean;
begin
  Result := FColIndex = HighCol;
end;

procedure TTableIterator.GoToCell(ACell: TCell);
begin
  FColIndex := ACell.Col;
  FRowIndex := ACell.Row;
end;

procedure TTableIterator.GoFirstRow;
begin
  FRowIndex := 0;
end;

procedure TTableIterator.GoNextRow;
begin
  if IsLastRow then begin
    { do nothing }
  end else begin
    Inc(FRowIndex);
  end;
end;

procedure TTableIterator.GoPreviousRow;
begin
  if IsFirstRow then begin
    { do nothing }
  end else begin
    Dec(FRowIndex);
  end;
end;

procedure TTableIterator.GoLastRow;
begin
  FRowIndex := HighRow;
end;

procedure TTableIterator.GoFirstCol;
begin
  FColIndex := 0;
end;

procedure TTableIterator.GoNextCol;
begin
  if IsLastCol then begin
    { do nothing }
  end else begin
    Inc(FColIndex);
  end;
end;

procedure TTableIterator.GoPreviousCol;
begin
  if IsFirstCol then begin
    { do nothing }
  end else begin
    Dec(FColIndex);
  end;
end;

procedure TTableIterator.GoLastCol;
begin
  FColIndex := HighCol;
end;

procedure TTableIterator.Save;
begin
  FSavedCol := FColIndex;
  FSavedRow := FRowIndex;
end;

procedure TTableIterator.Load;
begin
  FColIndex := FSavedCol;
  FRowIndex := FSavedRow;
end;

end.
