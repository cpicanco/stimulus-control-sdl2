{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Generics.Aggregator.Contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Classes, Generics.Collections, Generics.Iterator.Contract;

type

  { IAggregator }

  generic IAggregator<_GT> = interface
    function List: specialize TList<_GT>;
    function Iterator: specialize IIterator<_GT>;
  end;

implementation

end.

