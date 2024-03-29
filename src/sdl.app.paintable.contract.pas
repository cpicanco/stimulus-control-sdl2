{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.paintable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

type

  IPaintable = interface
    ['{600FC349-A1FA-44CA-986B-878F34F0533F}']
    //function Invalidated : Boolean;
    //procedure Invalidate;
    procedure Paint;
    //procedure Validate;
  end;

implementation

end.

