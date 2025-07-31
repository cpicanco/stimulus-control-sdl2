{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Generics.Iterator.Table.Contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Generics.Tables.Types;

type

  { ITableIterator }

  generic ITableIterator<_GT> = interface
    function GetCurrent : _GT;
    function IsCurrentEmpty(out ACell: _GT) : Boolean;
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

end.
