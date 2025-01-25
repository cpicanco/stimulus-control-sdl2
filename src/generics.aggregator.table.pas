{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit generics.aggregator.table;

{$mode ObjFPC}{$H+}

interface

uses SysUtils, Generics.Collections
   , Generics.Aggregator.Table.Contract
   , Generics.Iterator.Table.Contract
   , Generics.Iterator.Table
   , Generics.Table;

type

  { TTableAggregator }

  generic TTableAggregator<_GT> = class (specialize ITableAggregator<_GT>)
  public type
    TIteratorSpec = specialize TTableIterator<_GT>;
    TTableSpec  = specialize TTable<_GT>;
    IIteratorSpec = specialize ITableIterator<_GT>;
  private

  protected
    FTable : TTableSpec;
    FIterator : TIteratorSpec;
  public
    constructor Create; virtual;
    destructor Destroy; override;
    function Table: TTableSpec;
    function Iterator: IIteratorSpec;
  end;

implementation

{ TTableAggregator }

constructor TTableAggregator.Create;
begin
  FTable := TTableSpec.Create;
  FIterator:=TIteratorSpec.Create(Self);
end;

destructor TTableAggregator.Destroy;
begin
  FIterator.Free;
  FTable.Free;
  inherited Destroy;
end;

function TTableAggregator.Table: TTableSpec;
begin
  Result := FTable;
end;

function TTableAggregator.Iterator: IIteratorSpec;
begin
  Result := FIterator as IIteratorSpec;
end;


end.

