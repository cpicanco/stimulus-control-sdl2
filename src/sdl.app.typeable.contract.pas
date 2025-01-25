{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.typeable.contract;

{$mode ObjFPC}{$H+}

interface

uses
  SDL2, sdl.app.events.abstract;

type

  ITypeable = interface
    ['{A7B5AA4B-C570-47F5-B9BB-C47774851570}']
    function GetSDLKeyDown : TOnKeyDownEvent;
    function GetSDLKeyUp : TOnKeyUpEvent;
    function GetSDLTextInputEvent : TOnTextInputEvent;
    function GetSDLTextEditingEvent : TOnTextEditingEvent; // for onscreen text editing
    procedure SDLKeyDown(const event: TSDL_KeyboardEvent);
    procedure SDLKeyUp(const event: TSDL_KeyboardEvent);
    procedure SDLTextInput(const event: TSDL_TextInputEvent);
    procedure SDLTextEditingEvent(const event: TSDL_TextEditingEvent);// for onscreen text editing
    procedure KeyDown(Sender: TObject; Key: TSDL_KeyCode; Shift: TCustomShiftState);
    procedure KeyPress(Sender: TObject; var Key: char);
    procedure KeyUp(Sender: TObject; Key: TSDL_KeyCode; Shift: TCustomShiftState);
  end;

implementation

end.

