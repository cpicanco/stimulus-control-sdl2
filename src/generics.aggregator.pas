{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit generics.aggregator;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, Generics.Collections
   , Generics.Aggregator.Contract
   , Generics.Iterator.Contract
   , Generics.Iterator;

type

  { TAggregator }

  generic TAggregator<_GT> = class (specialize IAggregator<_GT>)
  public type
    TIteratorSpec = specialize TIterator<_GT>;
    TListSpec  = specialize TList<_GT>;
    IIteratorSpec = specialize IIterator<_GT>;
  private

  protected
    FList : TListSpec;
    FIterator : TIteratorSpec;
  public
    constructor Create;
    destructor Destroy; override;
    function List: TListSpec;
    function Iterator: IIteratorSpec;
  end;

implementation

{ TAggregator }

constructor TAggregator.Create;
begin
  FList := TListSpec.Create;
  FIterator:=TIteratorSpec.Create(Self);
end;

destructor TAggregator.Destroy;
begin
  FIterator.Free;
  FList.Free;
  inherited Destroy;
end;

function TAggregator.List: TListSpec;
begin
  Result := FList;
end;

function TAggregator.Iterator: IIteratorSpec;
begin
  Result := FIterator as IIteratorSpec;
end;


end.

