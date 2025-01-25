{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Generics.Aggregator.Table.Contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Generics.Table, Generics.Iterator.Table.Contract;

type

  { IAggregator }

  generic ITableAggregator<_GT> = interface
    function Table: specialize TTable<_GT>;
    function Iterator: specialize ITableIterator<_GT>;
  end;

implementation

end.

