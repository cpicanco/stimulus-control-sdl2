{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.dragdropable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

type

  IDragDropable = interface
  ['{11E4209B-68FD-4866-80C6-CBF74E22F41C}']
    procedure DragDrop(Sender, Source: TObject; X, Y: Integer);
  end;

implementation

end.

