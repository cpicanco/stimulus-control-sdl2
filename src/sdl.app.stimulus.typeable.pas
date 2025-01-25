{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.typeable;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.stimulus
  , sdl.app.typeable.contract
  , sdl.app.events.abstract
  , sdl.app.events.custom
  , sdl.app.graphics.text
  {$IFDEF WINDOWS}
  , windows
  {$ENDIF}
  ;

type

  { TTypeableStimulus }

  TTypeableStimulus = class(TStimulus, ITypeable)
  private
    FRect : TSDL_Rect;
    FText : TText;
    FFocused: Boolean;
    FOnKeyDown: TKeyEvent;
    FOnKeyPress: TKeyPressEvent;
    FOnKeyUp: TKeyEvent;
    function GetSDLKeyDown : TOnKeyDownEvent;
    function GetSDLKeyUp : TOnKeyUpEvent;
    function GetSDLTextInputEvent : TOnTextInputEvent;
    function GetSDLTextEditingEvent: TOnTextEditingEvent;
    procedure SDLKeyDown(const event: TSDL_KeyboardEvent);
    procedure SDLKeyUp(const event: TSDL_KeyboardEvent);
    procedure SDLTextInput(const event: TSDL_TextInputEvent);
    procedure SDLTextEditingEvent(const event: TSDL_TextEditingEvent);
    procedure SetFocused(AValue: Boolean);
    procedure SetOnKeyDown(AValue: TKeyEvent);
    procedure SetOnKeyPress(AValue: TKeyPressEvent);
    procedure SetOnKeyUp(AValue: TKeyEvent);
  protected
    procedure KeyDown(Sender: TObject; Key: TSDL_KeyCode; Shift: TCustomShiftState); virtual;
    procedure KeyUp(Sender: TObject; Key: TSDL_KeyCode; Shift: TCustomShiftState); virtual;
    procedure KeyPress(Sender: TObject; var Key: char); virtual;
  public
    constructor Create; override;
    destructor Destroy; override;
    function AsTypeable : ITypeable;
    function IsCorrectResponse : Boolean; override;
    procedure Focus;
    procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); override;
    procedure Start; override;
    procedure Stop; override;
    property Focused : Boolean read FFocused write SetFocused;
    property OnKeyDown  : TKeyEvent read FOnKeyDown write SetOnKeyDown;
    property OnKeyUp    : TKeyEvent read FOnKeyUp write SetOnKeyUp;
    property OnKeyPress : TKeyPressEvent read FOnKeyPress write SetOnKeyPress;
  end;

implementation

uses sdl.app.controls.custom
   , session.strutils
   ;

{ TTypeableStimulus }

function TTypeableStimulus.AsTypeable: ITypeable;
begin
  Result := Self as ITypeable;
end;

procedure TTypeableStimulus.Focus;
begin
  Focused := True;
end;

function TTypeableStimulus.GetSDLKeyDown: TOnKeyDownEvent;
begin
  Result := @SDLKeyDown;
end;

function TTypeableStimulus.GetSDLKeyUp: TOnKeyUpEvent;
begin
  Result := @SDLKeyUp;
end;

function TTypeableStimulus.GetSDLTextInputEvent: TOnTextInputEvent;
begin
  Result := @SDLTextInput;
end;

function TTypeableStimulus.GetSDLTextEditingEvent: TOnTextEditingEvent;
begin
  Result := @SDLTextEditingEvent;
end;


procedure TTypeableStimulus.SDLKeyDown(const event: TSDL_KeyboardEvent);
var
  Shift : TCustomShiftState;
begin
  Shift := GetShiftState;
  // https://wiki.libsdl.org/SDL2/SDL_Keycode
  KeyDown(Self, event.keysym.sym, Shift);
end;

procedure TTypeableStimulus.SDLKeyUp(const event: TSDL_KeyboardEvent);
var
  Shift : TCustomShiftState;
begin
  Shift := GetShiftState;

  // https://wiki.libsdl.org/SDL2/SDL_Keycode
  KeyUp(Self, event.keysym.sym, Shift);
end;

procedure TTypeableStimulus.SDLTextInput(const event: TSDL_TextInputEvent);
begin

end;

procedure TTypeableStimulus.SDLTextEditingEvent(
  const event: TSDL_TextEditingEvent);
begin
  //event.text
  //if Assigned(FText) then begin
    //FText.Visible := False;
    //FText.Clear;
    //FText.Load(ArrayToStr(event.text, event.length), 'Raleway-Regular');
    //FText.Visible := True;
  //end;
end;

procedure TTypeableStimulus.SetFocused(AValue: Boolean);
begin
  if FFocused = AValue then Exit;
  FFocused := AValue;
end;

procedure TTypeableStimulus.SetOnKeyDown(AValue: TKeyEvent);
begin
  if FOnKeyDown = AValue then Exit;
  FOnKeyDown := AValue;
end;

procedure TTypeableStimulus.SetOnKeyPress(AValue: TKeyPressEvent);
begin
  if FOnKeyPress = AValue then Exit;
  FOnKeyPress := AValue;
end;

procedure TTypeableStimulus.SetOnKeyUp(AValue: TKeyEvent);
begin
  if FOnKeyUp = AValue then Exit;
  FOnKeyUp := AValue;
end;

procedure TTypeableStimulus.KeyDown(Sender: TObject; Key: TSDL_KeyCode;
  Shift: TCustomShiftState);
begin

end;

procedure TTypeableStimulus.KeyUp(Sender: TObject; Key: TSDL_KeyCode;
  Shift: TCustomShiftState);
begin

end;

procedure TTypeableStimulus.KeyPress(Sender: TObject; var Key: char);
begin

end;

constructor TTypeableStimulus.Create;
begin
  inherited Create;
  FText := TText.Create;
  FText.Owner := Self;
  //EventHandler.OnKeyDown := AsTypeable.GetSDLKeyDown;
  SDLEvents.OnKeyUp := AsTypeable.GetSDLKeyUp;
  SDLEvents.OnTextInput := AsTypeable.GetSDLTextInputEvent;
  //EventHandler.OnTextEditing := AsTypeable.GetSDLTextEditingEvent;
end;

destructor TTypeableStimulus.Destroy;
begin
  FText.Free;
  SDLEvents.OnKeyUp := nil;
  SDLEvents.OnKeyDown := nil;
  SDLEvents.OnTextInput := nil;
  SDLEvents.OnTextEditing := nil;
  inherited Destroy;
end;

function TTypeableStimulus.IsCorrectResponse : Boolean;
begin
  //Result := LowerCase(FTextFromVocalResponse) = FWord;
end;

procedure TTypeableStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
const
  LKey : PAnsiChar = SDL_HINT_IME_SHOW_UI;
  LValue : PAnsiChar = '1';
begin
  FRect := ARect;
  with FRect do begin
    y := y - h;
    h := h div 2;
    w := w * 2;
  end;
  FText.Load('teste');
  FText.CentralizeWith(FRect);
  FText.Parent := TSDLControl(AParent);
  {$IFDEF WINDOWS}
    //SetEnvironmentVariable(LKey, LValue);
  {$ENDIF}
  //SDL_SetTextInputRect(@FRect);
  //FWord := GetWordValue(AParameters, IsSample, Index);
end;

procedure TTypeableStimulus.Start;
begin
  FText.Show;
end;

procedure TTypeableStimulus.Stop;
begin
  FText.Hide;
end;


end.

