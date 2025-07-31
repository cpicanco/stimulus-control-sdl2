{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.button;

interface

uses
  Classes, SysUtils
  //, SDL2
  , sdl.app.graphics.picture
  , sdl.app.events.abstract;

type
  { TButton }

  TButton = class(TPicture)
  private
    FSender : TObject;
    FOnClick: TNotifyEvent;
  protected
    FIsPressed: Boolean;
    procedure MouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure MouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure Click; virtual;
    procedure ShrinkHeight;
    property OnClick: TNotifyEvent read FOnClick write FOnClick;
    property Sender : TObject read FSender write FSender;
  end;

implementation

//uses sdl.app.video.methods, sdl.colors;

{ TButton }

constructor TButton.Create;
begin
  inherited Create;
  FIsPressed := False;
  FSender := nil;
end;

destructor TButton.Destroy;
begin
  inherited Destroy;
end;

procedure TButton.MouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
begin
  if Visible then begin
    inherited MouseDown(Sender, Shift, X, Y);
    FIsPressed := True;

  end;
end;

procedure TButton.MouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
begin
  if Visible then begin
    inherited MouseUp(Sender, Shift, X, Y);
    FIsPressed := False;
    if Assigned(OnClick) then
      OnClick(Self);
  end;
end;

procedure TButton.Paint;
begin
  inherited Paint;

end;

procedure TButton.Click;
begin
  if Assigned(FOnClick) then
    FOnClick(Self);
end;

procedure TButton.ShrinkHeight;
begin
  Height:= Height div 3;
end;

end.
